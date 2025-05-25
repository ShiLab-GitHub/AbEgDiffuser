#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_min, scatter_mean
from AbEgDiffuser.utils.protein.constants import BBHeavyAtom, atomsymb_to_atomindex, atom_unk, possymb_to_posindex, atom_pos_unk

class BlockEmbedding(nn.Module):
    '''
    [atom embedding + block embedding + atom position embedding]
    '''

    def __init__(self, embed_size, max_num_atoms=15, num_atom_type=6, num_atom_position=11, no_block_embedding=False):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.embed_size = embed_size
        self.no_block_embedding = no_block_embedding
        self.atom_embedding = nn.Embedding(num_atom_type, embed_size, padding_idx=atomsymb_to_atomindex['PAD'])
        self.position_embedding = nn.Embedding(num_atom_position, embed_size, padding_idx=possymb_to_posindex['PAD'])


    def forward(self, res_feat, atom_types, atom_positions, mask_atoms, block_lengths, structure_mask=None, sequence_mask=None):
        """
        Args:
            aa:         (N, L).
            atom_types: (N, L, A).
            atom_positions: (N, L, A).
            mask_atoms: (N, L, A).
            block_lengths:  (N, L).
            structure_mask: (N, L), mask out unknown structures to generate.
            sequence_mask:  (N, L), mask out unknown amino acids to generate.

        B: [Nb], block (residue) types
        A: [Nu], unit (atom) types
        atom_positions: [Nu], unit (atom) position encoding
        block_id: [Nu], block id of each unit
        """
        # Remove other atoms
        atom_types = atom_types[:, :, :self.max_num_atoms]
        atom_positions = atom_positions[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]
        block_lengths = torch.ones_like(block_lengths, dtype=torch.int)*self.max_num_atoms

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # (N, L)
        mask_residue_atom = mask_residue[:, :, None].expand(atom_types.shape) # (N, L, max_num_atoms)

        if sequence_mask is not None:
            # Avoid data leakage at training time
            atom_types = torch.where(sequence_mask[:, :, None].expand(atom_types.shape), atom_types, torch.full_like(atom_types, fill_value=int(atomsymb_to_atomindex[atom_unk])))
            atom_positions = torch.where(sequence_mask[:, :, None].expand(atom_positions.shape), atom_positions, torch.full_like(atom_positions, fill_value=int(possymb_to_posindex[atom_pos_unk])))

        res_feat = res_feat.reshape(-1, self.embed_size)
        atom_types = atom_types.flatten()
        atom_positions = atom_positions.flatten()
        block_lengths = block_lengths.flatten()
        mask = mask_residue_atom.flatten()

        with torch.no_grad():
            block_ids = torch.zeros_like(atom_types)  # [Nu]
            block_ids[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_ids.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

        atom_embed = self.atom_embedding(atom_types) + self.position_embedding(atom_positions)
        if self.no_block_embedding:
            out_embed = atom_embed
        else:
            block_embed = res_feat[block_ids]
            out_embed = atom_embed + block_embed


        out_embed = out_embed * mask[:, None] # [Nu, embed_size]

        return out_embed

