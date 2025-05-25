import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm as LayerNorm

from esm.pretrained import load_model_and_alphabet_local
from AbEgDiffuser.utils.protein.constants import ressymb_to_resindex, Fragment

class Linear_common(nn.Linear):
    def __init__(self, input_dim, output_dim, init, bias=True):
        super().__init__(input_dim, output_dim, bias=bias)

        assert init in ['gate', 'final', 'attn', 'relu', 'linear']

        if init in ['gate', 'final']:
            nn.init.constant_(self.weight, 0.)
        elif init == 'attn':
            # GlorotUniform
            torch.nn.init.xavier_uniform_(self.weight)
        elif init in ['relu', 'linear']:
            # Relu, He
            # linear, Le cun
            distribution_stddev = 0.87962566103423978
            scale = 2. if init == 'relu' else 1.
            stddev = np.sqrt(scale / input_dim) / distribution_stddev
            nn.init.trunc_normal_(self.weight, mean=0., std=stddev)
        else:
            raise NotImplementedError(f'{init} not Implemented')

        if bias:
            if init == 'gate':
                nn.init.constant_(self.bias, 1.)
            else:
                nn.init.constant_(self.bias, 0.)

def Linear(input_dim, output_dim, init, bias=True, config=None):
    assert init in ['gate', 'final', 'attn', 'relu', 'linear']
    if config is not None:
        return Linear_common(input_dim, output_dim, init, bias)
    else:
        return Linear_common(input_dim, output_dim, init, bias)


def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None


class ESMEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.esm_embed

        self.sep_pad_num = self.config.sep_pad_num
        self.repr_layer = self.config.repr_layer

        self.model_path = self.config.model_path
        self.model, self.alphabet = load_model_and_alphabet_local(self.model_path)
        self.model.requires_grad_(False)
        self.model.half() #在支持半精度的设备上取消注释
        self.batch_converter = self.alphabet.get_batch_converter()

    def _make_one_antibody_seq(self, heavy_seq, light_seq, ag_seq):

        str_heavy_seq_ = ''.join([get_key_by_value(ressymb_to_resindex, index) for index in heavy_seq])
        str_light_seq_ = ''.join([get_key_by_value(ressymb_to_resindex, index) for index in light_seq])
        str_ag_seq_ = ''.join([get_key_by_value(ressymb_to_resindex, index) for index in ag_seq])

        return str_heavy_seq_, str_light_seq_, str_ag_seq_

    def extract(self, label_seqs, device, linker_mask=None):

        max_len = max([len(s) for l, s in label_seqs])

        if type(self.repr_layer) is int:
            self.repr_layer = [self.repr_layer]

        with torch.no_grad():
            batch_labels, batch_strs, batch_tokens = self.batch_converter(label_seqs)
            batch_tokens = batch_tokens.to(device=device)

            if linker_mask is not None:
                batch_tokens = torch.where(linker_mask, self.alphabet.padding_idx, batch_tokens)

            results = self.model(batch_tokens, repr_layers=self.repr_layer)
            single = [results['representations'][r][:, 1: 1 + max_len] for r in self.repr_layer]

            ret = dict(single=single)

        return ret

    def forward(self, s_t, fragment_type):
        # seq_t, antibody_len = batch['seq_t'], batch['anchor_flag'].shape[1]
        N, L = s_t.shape
        ab_heavy = [s_t[i][fragment_type[i] == Fragment.Heavy] for i in range(N)]
        ab_light = [s_t[i][fragment_type[i] == Fragment.Light] for i in range(N)]
        ag = [s_t[i][fragment_type[i] == Fragment.Antigen] for i in range(N)]

        str_seq_t = [self._make_one_antibody_seq(x, y, z) for x, y, z in
                     zip(ab_heavy, ab_light, ag)]

        str_heavy_seq, str_light_seq, str_ag_seq = zip(*str_seq_t)

        linker_mask = None
        if self.sep_pad_num == 0:
            heavy_embed = self.extract(list(zip(torch.tensor([i for i in range(s_t.shape[0])]), str_heavy_seq)), linker_mask=linker_mask, device=s_t.device)
            light_embed = self.extract(list(zip(torch.tensor([i for i in range(s_t.shape[0])]), str_light_seq)), linker_mask=linker_mask, device=s_t.device)
            ag_embed = self.extract(list(zip(torch.tensor([i for i in range(s_t.shape[0])]), str_ag_seq)), linker_mask=linker_mask, device=s_t.device)

            embed = [torch.cat([heavy_embed['single'][k, :len(x)], light_embed['single'][k, :len(y)], ag_embed['single'][k, :len(z)]], dim=0) for
                     k, (x, y, z) in enumerate(zip(str_heavy_seq, str_light_seq, str_ag_seq))]
        else:
            sep_pad_seq = [h + 'G' * self.sep_pad_num + l + 'G' * self.sep_pad_num + g for h, l, g in
                                    zip(str_heavy_seq, str_light_seq, str_ag_seq)]

            embed = self.extract(list(zip(torch.tensor([i for i in range(s_t.shape[0])]), sep_pad_seq)), linker_mask=linker_mask, device=s_t.device)

            if len(embed['single']) == 1:
                embed = embed['single'][0]
            else:
                embed = torch.stack(embed['single'], dim=-1)

            embed = [torch.cat([
                embed[k, :len(x)],
                embed[k, len(x) + self.sep_pad_num : len(x) + self.sep_pad_num + len(y)],
                embed[k, len(x) + self.sep_pad_num + len(y) + self.sep_pad_num : len(x) + self.sep_pad_num + len(y) + self.sep_pad_num + len(z)]],
                dim=0) for k, (x, y, z) in enumerate(zip(str_heavy_seq, str_light_seq, str_ag_seq))]

        # pad_for_batch
        esm_embed = []
        for item in embed:
            shape = [L - item.shape[0]] + list(item.shape[1:])
            z = torch.zeros(shape, dtype=item.dtype, device=item.device)
            c = torch.cat((item, z), dim=0)
            esm_embed.append(c)
        esm_embed = torch.stack(esm_embed, dim=0)

        return esm_embed

if __name__ == '__main__':


    from torch.nn import functional as F
    from easydict import EasyDict

    c = {"node_feat_dim": 128,
         "esm": {
             "enabled": True,
             "embed_dim": 2560,
             "num_layers": 36,
             "dropout_rate": 0.1,
             "norm": True,
             "esm_embed": {
                 "return_attnw": False,
                 "sep_pad_num": 48,
                 "repr_layer": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
                 "model_path": "/media/zz/新加卷/zyb/esm_model/esm2_t36_3B_UR50D.pt"
             }
         }}

    cfg = EasyDict(c)

    batch = {'aa': torch.tensor([[ 5, 15,  9,  3,  3, 15,  9, 16,  9, 16,  1, 16, 17, 15,  5,  7,  2, 12,
         11, 15,  2,  6, 10, 15, 18, 17, 14,  9,  3, 18,  7,  0,  7,  7, 19,  0,
         15,  5, 16, 16, 19, 19,  0, 15, 18,  0,  8,  5, 14,  4, 16,  7, 15,  8,
         16, 15, 16, 16, 17,  2,  9, 14,  7,  0, 15, 19,  4,  1,  0, 16, 19, 12,
         11, 19, 12, 16,  2, 11,  9, 18,  5,  5, 16,  5, 13, 17, 19, 11,  9,  9,
          9,  5, 15, 19,  2,  5, 11, 15,  0,  2,  1,  9,  0,  4,  5, 15,  4, 17,
          1,  4,  3,  6,  8, 15,  4,  2,  7, 15, 13,  1, 12,  8,  7,  5,  5,  6,
          5, 15,  8,  8,  1, 16,  5,  2,  0,  0,  4,  1, 15,  0, 19,  3,  1, 16,
          0, 13, 19,  0, 11,  0, 19,  1, 15,  6,  0, 11,  5, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 21],
        [13,  9, 17, 13, 15, 17,  8, 17, 15,  1,  8,  0, 16,  4, 15, 15,  9,  0,
          7, 15, 18, 17, 14, 13,  9,  3, 18, 10,  5,  5,  7,  7, 12,  7,  4,  5,
         16,  0, 11, 19,  0, 13,  8,  4, 13,  5, 14, 17, 16,  7, 16,  0,  2,  3,
         16, 15, 16,  0, 19, 10,  3,  9, 15,  2,  0, 17, 19, 19,  1,  0, 14,  5,
          5, 15, 17, 15,  5, 16,  9, 17,  2,  4,  2,  7, 18,  5, 13,  5, 16,  2,
          7, 18,  9,  0,  0, 12,  8,  9, 19, 13, 13, 19, 11,  7, 19, 12,  7, 16,
          4,  3,  9,  1,  2,  2,  2, 12, 12,  3,  7, 12,  6,  0, 16,  4,  8,  0,
         10,  0, 19,  8,  3,  5, 16, 10,  9, 11,  1,  3,  1,  8, 14,  5,  4, 14,
         14,  7, 15,  9, 19, 10,  9,  1, 16,  5, 11, 15, 15,  6, 15, 15, 18,  2,
         11, 13,  1, 13,  1, 10, 13, 12, 17,  2, 13,  0, 15,  9, 12,  5,  6,  1,
         14,  3, 12, 12, 12, 18,  3, 11,  3,  0, 16,  3, 14,  7, 19,  6,  4, 17,
         17,  5, 13, 10, 17, 19, 19, 13,  1, 17, 13,  5, 19, 14,  0,  9,  6, 14,
          5, 12,  0,  3, 15, 17,  1,  8, 10, 16, 16, 14, 18, 16, 13, 12, 13,  9,
          7,  1, 16, 21, 21, 21]]),'fragment_type': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0]]),'mask': torch.tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True, False, False, False]])}

    s_t = torch.tensor([[ 5, 15,  9,  3,  3, 15,  9, 16,  9, 16,  1, 16, 17, 15,  5,  7,  2, 12,
         11, 15,  2,  6, 10, 15, 18, 17, 14,  9,  3, 18,  7,  0,  7,  7, 19,  0,
         15,  5, 16, 16, 19, 19,  1, 1, 1,  1,  1,  1, 14,  4, 16,  7, 15,  8,
         16, 15, 16, 16, 17,  2,  9, 14,  7,  0, 15, 19,  4,  1,  0, 16, 19, 12,
         11, 19, 12, 16,  2, 11,  9, 18,  5,  5, 16,  5, 13, 17, 19, 11,  9,  9,
          9,  5, 15, 19,  2,  5, 11, 15,  0,  2,  1,  9,  0,  4,  5, 15,  4, 17,
          1,  4,  3,  6,  8, 15,  4,  2,  7, 15, 13,  1, 12,  8,  7,  5,  5,  6,
          5, 15,  8,  8,  1, 16,  5,  2,  0,  0,  4,  1, 15,  0, 19,  3,  1, 16,
          0, 13, 19,  0, 11,  0, 19,  1, 15,  6,  0, 11,  5, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
         21, 21, 21, 21, 21, 21],
        [13,  9, 17, 13, 15, 17,  8, 17, 15,  1,  8,  0, 16,  4, 15, 15,  9,  0,
          7, 15, 18, 17, 14, 13,  9,  3, 18, 10,  5,  5,  7,  7, 12,  7,  4,  5,
         16,  0, 11, 19,  0, 13,  8,  4, 13,  5, 14, 17, 16,  7, 16,  0,  2,  3,
         16, 15, 16,  0, 19, 1,  1,  1, 1,  1,  1, 17, 19, 19,  1,  0, 14,  5,
          5, 15, 17, 15,  5, 16,  9, 17,  2,  4,  2,  7, 18,  5, 13,  5, 16,  2,
          7, 18,  9,  0,  0, 12,  8,  9, 19, 13, 13, 19, 11,  7, 19, 12,  7, 16,
          4,  3,  9,  1,  2,  2,  2, 12, 12,  3,  7, 12,  6,  0, 16,  4,  8,  0,
         10,  0, 19,  8,  3,  5, 16, 10,  9, 11,  1,  3,  1,  8, 14,  5,  4, 14,
         14,  7, 15,  9, 19, 10,  9,  1, 16,  5, 11, 15, 15,  6, 15, 15, 18,  2,
         11, 13,  1, 13,  1, 10, 13, 12, 17,  2, 13,  0, 15,  9, 12,  5,  6,  1,
         14,  3, 12, 12, 12, 18,  3, 11,  3,  0, 16,  3, 14,  7, 19,  6,  4, 17,
         17,  5, 13, 10, 17, 19, 19, 13,  1, 17, 13,  5, 19, 14,  0,  9,  6, 14,
          5, 12,  0,  3, 15, 17,  1,  8, 10, 16, 16, 14, 18, 16, 13, 12, 13,  9,
          7,  1, 16, 21, 21, 21]])

    proj_aa_type = nn.Embedding(25, cfg.node_feat_dim, padding_idx=21)
    current_sequence_embedding = nn.Embedding(25, cfg.node_feat_dim)

    antibody_seq_act = proj_aa_type(s_t)
    print(f"antibody_seq_act = {antibody_seq_act.shape}")

    if cfg.esm.enabled:
        encode_esm_emb = ESMEmbedding(cfg.esm)
        esm_embed_weights = torch.zeros((cfg.esm.num_layers + 1,))
        esm_embed_weights = nn.Parameter(esm_embed_weights)

        proj_esm_embed = nn.Sequential(
            LayerNorm(cfg.esm.embed_dim),
            Linear(cfg.esm.embed_dim, cfg.node_feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(cfg.node_feat_dim, cfg.node_feat_dim, init='linear', bias=True),
        )

    if cfg.esm.enabled:
        layer_weights = F.softmax(esm_embed_weights, dim=-1)

        antibody_esm_embed = encode_esm_emb(s_t, batch['fragment_type']).to(dtype=layer_weights.dtype)
        antibody_esm_embed = torch.einsum('b l c n, n -> b l c', antibody_esm_embed, layer_weights)
        antibody_esm_embed = proj_esm_embed(antibody_esm_embed)
        antibody_seq_act = antibody_seq_act + antibody_esm_embed
        print(f"antibody_seq_act = {antibody_seq_act.shape}")


