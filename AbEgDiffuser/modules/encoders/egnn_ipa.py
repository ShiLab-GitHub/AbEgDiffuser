import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_min, scatter_mean

from AbEgDiffuser.modules.encoders.get import GET
from AbEgDiffuser.modules.encoders.ipa import GAEncoder
from AbEgDiffuser.utils.protein.constants import BBHeavyAtom

class EGNNEncoder(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, no_block=False, num_layers=4, get_block_opt={}):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.no_block = no_block
        self.encoder = GET(node_feat_dim, edge_feat_dim, num_layers, **get_block_opt)

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr, mask_generate, mask_atoms):

        H, Z = self.encoder(H, Z, block_id, batch_id, edges, edge_attr)

        N, L, num_atom = mask_atoms.shape
        if self.no_block:
            block_ids = torch.arange(0, N * L).to(mask_atoms.device)
            block_id = torch.repeat_interleave(block_ids, repeats=num_atom)

        mask = torch.where(mask_generate[:, :, None].expand_as(mask_atoms), torch.ones_like(mask_atoms, dtype=torch.bool), mask_atoms)
        # mask = mask_atoms
        res_repr = scatter_sum(H * (mask.flatten()[:, None]), block_id, dim=0)  # [Nb, hidden]
        res_repr = F.normalize(res_repr, dim=-1)
        batch_repr = res_repr.view(N, -1, self.node_feat_dim)  # [bs, hidden]
        batch_repr = F.normalize(batch_repr, dim=-1)

        H = batch_repr # (N, L, node_feat_dim)
        Z_global = Z.reshape(N, L, -1, 3) * mask[:, :, :, None]

        return H, Z_global

class EquivariantIPA_ED(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, max_num_atoms, no_block=False, enc_num_layers=4, dec_num_layers=4, eps_net_opt={}):
        super().__init__()
        self.encoder = EGNNEncoder(node_feat_dim, edge_feat_dim, no_block, enc_num_layers, **eps_net_opt)
        self.decoder = GAEncoder(node_feat_dim, edge_feat_dim, dec_num_layers, **eps_net_opt)

    def forward(self, R_t, p_t, H, Z, block_id, batch_id, edges, edge_attr, mask_generate, mask_atoms, pair_feat):
        H, Z_global = self.encoder(H, Z, block_id, batch_id, edges, edge_attr, mask_generate, mask_atoms)

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # (N, L)
        H_mixer = H * mask_residue[:, :, None]

        res_feat = self.decoder(R_t, p_t, H_mixer, pair_feat, mask_residue)

        return res_feat

