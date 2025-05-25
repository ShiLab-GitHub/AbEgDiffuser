import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm

from AbEgDiffuser.modules.common.geometry import reconstruct_backbone, apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix
from AbEgDiffuser.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from AbEgDiffuser.modules.encoders.esm_embed import ESMEmbedding, Linear
from AbEgDiffuser.modules.encoders.egnn_ipa import EquivariantIPA_ED
from AbEgDiffuser.modules.diffusion.transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition
from AbEgDiffuser.utils.protein.constants import max_num_heavyatoms, AA, BBHeavyAtom, restype_to_heavyatom_names, atom_pad, atomsymb_to_atomindex, atom_pos_pad, possymb_to_posindex

from AbEgDiffuser.utils.protein.parsers import _get_pos_code
from AbEgDiffuser.modules.common.nn_utils import init_mask
from AbEgDiffuser.modules.encoders.edge import EdgeEmbedding

def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)

    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )

    loss = loss.reshape(size + [3]).sum(dim=-1)    # (*, )
    return loss


def reconstruct_noised_coord(coord, r_noisy, p_noisy, s_noisy, chain_nb, res_nb, mask_atoms, mask_gen):
    """
    Args:
        coord: (N, L, max_num_atom, 3)
        p_noisy: (N, L, 3)
        v_noisy: (N, L, 3)
        mask_gen: (N, L) bool, true denotes the residue to be generated
    """

    mask_res = mask_atoms[:, :, BBHeavyAtom.CA]
    pos_recons = reconstruct_backbone(r_noisy, p_noisy, s_noisy, chain_nb, res_nb, mask_res)  # (N, L, 4, 3)
    pad_pos_recons = coord.clone()   # (N, L, A, 3)
    pad_pos_recons[:,:,:4] = pos_recons

    new_coord = torch.where(
        mask_gen[:, :, None, None].expand_as(coord),
        pad_pos_recons, coord
    )
    return new_coord

def get_residue_heavyatom_info(s_noisy):
    N, L = s_noisy.shape
    mask_heavyatom = torch.zeros([N, L, max_num_heavyatoms], dtype=torch.bool).to(s_noisy.device)
    atom_types = torch.full_like(mask_heavyatom, atomsymb_to_atomindex[atom_pad], dtype=torch.long).to(s_noisy.device)
    atom_positions = torch.full_like(mask_heavyatom, possymb_to_posindex[atom_pos_pad], dtype=torch.long).to(s_noisy.device)

    for i in range(N):
        for j in range(L):
            for idx, atom_name in enumerate(restype_to_heavyatom_names[s_noisy[i][j].item()]):
                if atom_name == '': continue
                mask_heavyatom[i][j][idx] = True
                atom_types[i][j][idx] = torch.tensor(atomsymb_to_atomindex[atom_name[0]])
                atom_positions[i][j][idx] = torch.tensor(possymb_to_posindex[_get_pos_code(atom_name, atom_name[0])])
    return mask_heavyatom, atom_types, atom_positions

class EpsilonNet(nn.Module):

    def __init__(self, node_feat_dim, edge_feat_dim, num_atoms, cfg_esm, no_block, e_num_layers, i_num_layers, encoder_opt={}):
        super().__init__()

        self.cfg = cfg_esm

        if self.cfg.enabled:
            self.encode_esm_emb = ESMEmbedding(self.cfg)
            esm_embed_weights = torch.zeros((self.cfg.num_layers + 1,))
            self.esm_embed_weights = nn.Parameter(esm_embed_weights)
            self.proj_esm_embed = nn.Sequential(
                nn.LayerNorm(self.cfg.embed_dim),
                Linear(self.cfg.embed_dim, node_feat_dim, init='linear', bias=True),
                nn.ReLU(),
                Linear(node_feat_dim, node_feat_dim, init='linear', bias=True),
            )
        else:
            self.current_sequence_embedding = nn.Embedding(25, node_feat_dim, padding_idx=AA.PAD)  # 21 is padding

        self.res_feat_mixer = nn.Sequential(
            nn.Linear(node_feat_dim * 2, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim),
        )

        self.encoder = EquivariantIPA_ED(node_feat_dim, edge_feat_dim, num_atoms, no_block, e_num_layers, i_num_layers, **encoder_opt) # 等变图神经网络编码器，IPA解码器

        self.eps_crd_net = nn.Sequential(
            nn.Linear(node_feat_dim+3, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, 3)
        )

        self.eps_rot_net = nn.Sequential(
            nn.Linear(node_feat_dim+3, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, 3)
        )

        self.eps_seq_net = nn.Sequential(
            nn.Linear(node_feat_dim+3, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, 20), nn.Softmax(dim=-1)
        )


    def forward(self, v_t, p_t, s_t, fragment_type, H, Z, block_id, batch_id, edges, edge_attr, beta, mask_generate, mask_atoms, pair_feat):
        """
        Args:
            v_t:    (N, L, 3).
            p_t:    (N, L, 3).
            s_t:    (N, L).
            res_feat:   (N, L, res_dim).
            pair_feat:  (N, L, L, pair_dim).
            H_0:  (N_atom, node_feat_dim)
            pos_heavyatom:  (N_atom, 1, 3)
            edges:  (2, N_edge)
            edge_attr: (N_edge, edge_feat_dim)
            beta:   (N,).
            mask_generate:    (N, L).
            mask_res:       (N, L).
            mask_atoms:       (N, L, num_atoms).
        Returns:
            v_next: UPDATED (not epsilon) SO3-vector of orietnations, (N, L, 3).
            eps_pos: (N, L, 3).
        """
        N, L, num_atoms = mask_atoms.size()
        R_t = so3vec_to_rotation(v_t) # (N, L, 3, 3)

        # s_t = s_t.clamp(min=0, max=19)  # TODO: clamping is good but ugly.
        # antibody_seq_act = self.current_sequence_embedding(s_t)
        if self.cfg.enabled:
            layer_weights = F.softmax(self.esm_embed_weights, dim=-1)

            antibody_esm_embed = self.encode_esm_emb(s_t, fragment_type).to(dtype=layer_weights.dtype)
            antibody_esm_embed = torch.einsum('b l c n, n -> b l c', antibody_esm_embed, layer_weights)
            antibody_esm_embed = self.proj_esm_embed(antibody_esm_embed)
            antibody_seq_act = antibody_esm_embed
        else:
            antibody_seq_act = self.current_sequence_embedding(s_t)

        seq_act = antibody_seq_act[:, :, None, :].expand(-1, -1, num_atoms, -1)

        H = self.res_feat_mixer(torch.cat([H, seq_act.reshape(N*L*num_atoms, -1)], dim=-1)) # [Important] Incorporate sequence at the current step.

        res_feat = self.encoder(R_t, p_t, H, Z.reshape(-1, 1, 3), block_id, batch_id, edges, edge_attr, mask_generate, mask_atoms, pair_feat) # ((N, L, node_feat_dim)

        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(N, L, 3)

        in_feat = torch.cat([res_feat, t_embed], dim=-1)

        # Position changes
        eps_crd = self.eps_crd_net(in_feat)    # (N, L, 3)
        eps_pos = apply_rotation_to_vector(R_t, eps_crd)  # (N, L, 3)
        eps_pos = torch.where(mask_generate[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))

        # New orientation
        eps_rot = self.eps_rot_net(in_feat)    # (N, L, 3)
        U = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3)
        R_next = R_t @ U
        v_next = rotation_to_so3vec(R_next)     # (N, L, 3)
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)

        # New sequence categorical distributions
        c_denoised = self.eps_seq_net(in_feat)  # Already softmax-ed, (N, L, 20)

        return v_next, R_next, eps_pos, c_denoised


class FullDDPM(nn.Module):

    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        cfg_esm,
        k_neighbors,
        single_present,
        num_atoms,
        num_steps=100,
        eps_net_opt={},
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
    ):
        super().__init__()

        self.num_atoms = num_atoms
        self.single_present = single_present
        self.edge_embed = EdgeEmbedding(edge_feat_dim, num_atoms, k_neighbors, single_present)
        self.eps_net = EpsilonNet(node_feat_dim, edge_feat_dim, num_atoms, cfg_esm, no_block=single_present, **eps_net_opt) # 预测噪声的核心网络

        self.num_steps = num_steps
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p
    
    def GraphEmbedding(self, coord, v_noisy, p_noisy, s_noisy, atom_types, block_lengths, lengths, mask_generate, mask_heavyatom, chain_nb, res_nb, fragment_type, denoise_structure, denoise_sequence):


        if denoise_structure:
            coord = reconstruct_noised_coord(
                coord=coord,
                r_noisy=so3vec_to_rotation(v_noisy),
                p_noisy=p_noisy,
                s_noisy=s_noisy,
                chain_nb=chain_nb,
                res_nb=res_nb,
                mask_atoms=mask_heavyatom,
                mask_gen=mask_generate,
            )

        block_id, batch_id, edges, edge_attr = self.edge_embed(
            pos_heavyatom=coord,
            aa=s_noisy,
            atom_types=atom_types,
            mask_atoms=mask_heavyatom,
            block_lengths=block_lengths,
            lengths=lengths,
            fragment_type=fragment_type,
        )

        return coord, block_id, batch_id, edges, edge_attr

    def forward(self, v_0, p_0, s_0, H, coord, atom_types, block_lengths, lengths, pair_feat,
                mask_generate, mask_atoms, fragment_type, chain_nb, res_nb, denoise_structure, denoise_sequence, t=None):
        N, L = s_0.shape
        if t == None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        p_0 = self._normalize_position(p_0)
        R_0 = so3vec_to_rotation(v_0)

        if denoise_structure:
            # Add noise to rotation
            v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)
            # Add noise to positions
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)
            coord = init_mask(coord, ~mask_generate)
            coord = self._normalize_position(coord)
        else:
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            # Add noise to sequence
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0.clone()

        noised_coord, block_id, batch_id, edges, edge_attr = self.GraphEmbedding(
            coord, v_noisy, p_noisy, s_noisy,
            atom_types, block_lengths, lengths, mask_generate, mask_atoms, chain_nb, res_nb, fragment_type,
            denoise_structure,
            denoise_sequence
        )

        beta = self.trans_pos.var_sched.betas[t]
        v_pred, R_pred, eps_p_pred, c_denoised = self.eps_net(
            v_noisy, p_noisy, s_noisy, fragment_type, H, noised_coord, block_id, batch_id, edges, edge_attr, beta, mask_generate, mask_atoms, pair_feat
        )

        loss_dict = {}

        # # Rotation loss
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0)  # (N, L)
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['rot'] = loss_rot

        # Position loss
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)  # (N, L)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos

        # Sequence categorical loss
        post_true = self.trans_seq.posterior(s_noisy, s_0, t)
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8)
        kldiv = F.kl_div(
            input=log_post_pred,
            target=post_true,
            reduction='none',
            log_target=False
        ).sum(dim=-1)    # (N, L)
        loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq

        return loss_dict

    @torch.no_grad()
    def sample(
        self,
        v, p, s,
        H, coord, atom_types, block_lengths, lengths, pair_feat,
        mask_generate, mask_atoms, fragment_type, chain_nb, res_nb,
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = s.shape[:2]
        p = self._normalize_position(p)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
            coord = init_mask(coord, ~mask_generate)
            coord = self._normalize_position(coord)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            noised_coord, block_id, batch_id, edges, edge_attr = self.GraphEmbedding(
                coord, v_t, p_t, s_t,
                atom_types, block_lengths, lengths, mask_generate, mask_atoms, chain_nb, res_nb, fragment_type,
                sample_structure,
                sample_sequence
            )

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, fragment_type, H, noised_coord, block_id, batch_id, edges, edge_attr, beta, mask_generate, mask_atoms, pair_feat
            )

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj

    @torch.no_grad()
    def optimize(
        self, 
        v, p, s,
        opt_step: int,
        H, coord, atom_types, block_lengths, lengths, pair_feat,
        mask_generate, mask_atoms, fragment_type, chain_nb, res_nb,
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Description:
            First adds noise to the given structure, then denoises it.
        """
        N, L = s.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            # Add noise to rotation
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t)
            # Add noise to positions
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
            coord = init_mask(coord, ~mask_generate)
            coord = self._normalize_position(coord)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, mask_generate, t)
            s_init = torch.where(mask_generate, s_noisy, s)
        else:
            s_init = s

        traj = {opt_step: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            noised_coord, block_id, batch_id, edges, edge_attr = self.GraphEmbedding(
                coord, v_t, p_t, s_t,
                atom_types, block_lengths, lengths, mask_generate, mask_atoms, chain_nb, res_nb, fragment_type,
                sample_structure,
                sample_sequence
            )

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, fragment_type, H, noised_coord, block_id, batch_id, edges, edge_attr, beta, mask_generate, mask_atoms, pair_feat
            )

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj
