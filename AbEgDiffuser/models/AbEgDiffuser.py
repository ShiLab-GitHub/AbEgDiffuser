import torch
import torch.nn as nn

from AbEgDiffuser.modules.common.geometry import construct_3d_basis
from AbEgDiffuser.modules.common.so3 import rotation_to_so3vec
from AbEgDiffuser.modules.encoders.node import BlockEmbedding
from AbEgDiffuser.modules.encoders.residue import ResidueEmbedding
from AbEgDiffuser.modules.encoders.pair import PairEmbedding
from AbEgDiffuser.modules.diffusion.ddpm_full import FullDDPM
from AbEgDiffuser.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom, num_aa_types, num_atom_types, num_atom_pos_types
from ._base import register_model


resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms,
    'backbone': 4
}


@register_model('AbEgDiffuser')
class DiffusionAntibodyDesign(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = ResidueEmbedding(cfg.node_feat_dim, self.num_atoms, num_aa_types)
        self.pair_embed = PairEmbedding(cfg.edge_feat_dim, self.num_atoms, num_aa_types)
        self.block_embed = BlockEmbedding(cfg.node_feat_dim, self.num_atoms, num_atom_types, num_atom_pos_types, no_block_embedding=cfg.single_present)

        self.diffusion = FullDDPM(
            cfg.node_feat_dim,
            cfg.edge_feat_dim,
            cfg.esm,
            cfg.k_neighbors,
            cfg.single_present,
            self.num_atoms,
            **cfg.diffusion,
        )

    def condition_encode(self, batch, remove_structure, remove_sequence):
        """
        Returns:
            res_feat:   (N, L, res_feat_dim)
            pair_feat:  (N, L, L, pair_feat_dim)
        """
        # This is used throughout embedding and encoding layers
        #   to avoid data leakage.
        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA],
            ~batch['generate_flag']     # Context means ``not generated''
        )

        structure_mask = context_mask if remove_structure else None
        sequence_mask = context_mask if remove_sequence else None

        res_feat = self.residue_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            fragment_type = batch['fragment_type'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )

        H = self.block_embed(
            res_feat,
            atom_types=batch['atom_types'],
            atom_positions=batch['atom_positions'],
            mask_atoms=batch['mask_heavyatom'],
            block_lengths=batch['block_lengths'],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )

        pair_feat = self.pair_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )

        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA] # CA 原子位置

        return H, pair_feat, R, p
    
    def forward(self, batch):
        mask_generate = batch['generate_flag']
        mask_atoms = batch['mask_heavyatom'][:, :, :self.num_atoms]
        fragment_type = batch['fragment_type']
        chain_nb = batch['chain_nb']
        res_nb = batch['res_nb']
        coord = batch['pos_heavyatom'][:, :, :self.num_atoms]
        atom_types = batch['atom_types'][:, :, :self.num_atoms]
        block_lengths = batch['block_lengths']
        lengths = batch['lengths']

        H, pair_feat, R_0, p_0 = self.condition_encode(
            batch,
            remove_structure=self.cfg.get('train_structure', True),
            remove_sequence=self.cfg.get('train_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)  # 旋转矩阵编码为 SO(3) 向量
        s_0 = batch['aa']

        loss_dict = self.diffusion(
            v_0, p_0, s_0, H, coord, atom_types, block_lengths, lengths, pair_feat,
            mask_generate, mask_atoms, fragment_type, chain_nb, res_nb,
            denoise_structure = self.cfg.get('train_structure', True),
            denoise_sequence = self.cfg.get('train_sequence', True),
        )
        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_atoms = batch['mask_heavyatom'][:, :, :self.num_atoms]
        fragment_type = batch['fragment_type']
        chain_nb = batch['chain_nb']
        res_nb = batch['res_nb']
        coord = batch['pos_heavyatom'][:, :, :self.num_atoms]
        atom_types = batch['atom_types'][:, :, :self.num_atoms]
        block_lengths = batch['block_lengths']
        lengths = batch['lengths']

        H, pair_feat, R_0, p_0 = self.condition_encode(
            batch,
            remove_structure=sample_opt.get('sample_structure', True),
            remove_sequence=sample_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        traj = self.diffusion.sample(v_0, p_0, s_0, H, coord, atom_types, block_lengths, lengths, pair_feat,
                                     mask_generate, mask_atoms, fragment_type, chain_nb, res_nb, **sample_opt)
        return traj

    @torch.no_grad()
    def optimize(
        self, 
        batch, 
        opt_step, 
        optimize_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_atoms = batch['mask_heavyatom'][:, :, :self.num_atoms]
        fragment_type = batch['fragment_type']
        chain_nb = batch['chain_nb']
        res_nb = batch['res_nb']
        coord = batch['pos_heavyatom'][:, :, :self.num_atoms]
        atom_types = batch['atom_types'][:, :, :self.num_atoms]
        block_lengths = batch['block_lengths']
        lengths = batch['lengths']

        H, pair_feat, R_0, p_0 = self.condition_encode(
            batch,
            remove_structure=optimize_opt.get('sample_structure', True),
            remove_sequence=optimize_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        traj = self.diffusion.optimize(v_0, p_0, s_0, opt_step, H, coord, atom_types, block_lengths, lengths, pair_feat,
                                       mask_generate, mask_atoms, fragment_type, chain_nb, res_nb, **optimize_opt)
        return traj
