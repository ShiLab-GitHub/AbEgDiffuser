#!/usr/bin/python
# -*- coding:utf-8 -*-
from itertools import groupby
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_min, scatter_mean

from AbEgDiffuser.utils.protein.constants import BBHeavyAtom, AA, atomsymb_to_atomindex, atom_unk
from AbEgDiffuser.modules.common.nn_utils import _unit_edges_from_block_edges, _knn_edges, BatchEdgeConstructor
from AbEgDiffuser.modules.common.nn_utils import init_mask

def pair2edge(edge_idx, lengths, pair):
    """
    convert pair to edges.

    Args:
        edge_idx (torch.Tensor): [2, N]
        pair (torch.Tensor): [B, L, L, D]
        lengths (torch.Tensor): [B, ]
    Returns:
        edge (torch.Tensor): [N, D]
    """
    # [B+1, ] the offsets of each sample in the batch, pad zero at the beginning.
    B = len(lengths)
    offsets = F.pad(torch.cumsum(lengths, dim=0), pad=(1, 0), value=0)
    edge_feats = []
    for i in range(B):
        prev_offset = offsets[i]
        next_offset = offsets[i+1]
        idx = (edge_idx >= prev_offset) & (edge_idx < next_offset)
        local_edge_idx = edge_idx[idx].reshape(2,-1) - prev_offset
        edge_feat = pair[i][local_edge_idx[0],local_edge_idx[1]]
        edge_feats.append(edge_feat)
    edge_feats = torch.cat(edge_feats,dim=0)
    return edge_feats

def _block_edge_dist(X, block_id, src_dst, mask_atoms_id): # , generate_atoms_id):
    '''
    Several units constitute a block.
    This function calculates the distance of edges between blocks
    The distance between two blocks are defined as the minimum distance of unit-pairs between them.
    The distance between two units are defined as the minimum distance across different channels.
        e.g. For number of channels c = 2, suppose their distance is c1 and c2, then the distance between the two units is min(c1, c2)

    :param X: [N, c, 3], coordinates, each unit has c channels. Assume the units in the same block are aranged sequentially
    :param block_id [N], id of block of each unit. Assume X is sorted so that block_id starts from 0 to Nb - 1
    :param src_dst: [Eb, 2], all edges (block level) that needs distance calculation, represented in (src, dst)
    '''
    (unit_src, unit_dst), (edge_id, _, _) = _unit_edges_from_block_edges(block_id, src_dst)

    # # 只保留src和dst都不是 ge_atom 的边, 其他边的dist设为0
    # ge_src = torch.isin(unit_src, generate_atoms_id)
    # ge_dst = torch.isin(unit_dst, generate_atoms_id)
    # keep_ge = torch.logical_and(ge_src, ge_dst)

    # 只保留src和dst都不是 mask_atom 的边, 其他边的dist设为无穷大
    mask_src = torch.isin(unit_src, mask_atoms_id)
    mask_dst = torch.isin(unit_dst, mask_atoms_id)
    keep_mask = torch.logical_and(mask_src, mask_dst)

    # calculate unit-pair distances
    src_x, dst_x = X[unit_src], X[unit_dst]  # [Eu, k, 3]
    dist = torch.norm(src_x - dst_x, dim=-1)  # [Eu, k]
    # dist = torch.where(keep_ge[:, None], dist, torch.zeros_like(dist))
    dist = torch.where(keep_mask[:, None], dist, torch.full_like(dist, fill_value=torch.inf))
    dist = torch.min(dist, dim=-1).values  # [Eu]
    dist = scatter_min(dist, edge_id)[0]  # [Eb]

    return dist


class KNNBatchEdgeConstructor(BatchEdgeConstructor):
    def __init__(self, k_neighbors, global_message_passing=True, global_node_id_vocab=[],
                 delete_self_loop=True) -> None:
        super().__init__(global_node_id_vocab, delete_self_loop)
        self.k_neighbors = k_neighbors
        self.global_message_passing = global_message_passing

    def _construct_intra_edges(self, S, batch_id, segment_ids, **kwargs):
        all_intra_edges = super()._construct_intra_edges(S, batch_id, segment_ids)
        X, block_id = kwargs['X'], kwargs['block_id']
        mask_atoms_id = kwargs['mask_atoms_id']
        # generate_atoms_id = kwargs['generate_atoms_id']
        # knn
        src_dst = all_intra_edges.T
        dist = _block_edge_dist(X, block_id, src_dst, mask_atoms_id) # , generate_atoms_id)
        intra_edges = _knn_edges(
            dist, src_dst, self.k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return intra_edges

    def _construct_inter_edges(self, S, batch_id, segment_ids, **kwargs):
        all_inter_edges = super()._construct_inter_edges(S, batch_id, segment_ids)
        X, block_id = kwargs['X'], kwargs['block_id']
        mask_atoms_id = kwargs['mask_atoms_id']
        # generate_atoms_id = kwargs['generate_atoms_id']
        # knn
        src_dst = all_inter_edges.T
        dist = _block_edge_dist(X, block_id, src_dst, mask_atoms_id) # , generate_atoms_id)
        inter_edges = _knn_edges(
            dist, src_dst, self.k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return inter_edges

    def _construct_global_edges(self, S, batch_id, segment_ids, **kwargs):
        if self.global_message_passing:
            return super()._construct_global_edges(S, batch_id, segment_ids, **kwargs)
        else:
            return None, None

    def _construct_seq_edges(self, S, batch_id, segment_ids, **kwargs):
        return None

class EdgeEmbedding(nn.Module):
    def __init__(self, edge_size=16, max_num_atoms=15, k_neighbors=9, no_block=False, global_message_passing=True, global_node_id_vocab=[], delete_self_loop=True):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.no_block = no_block
        self.edge_constructor = KNNBatchEdgeConstructor(k_neighbors=k_neighbors, global_message_passing=global_message_passing, global_node_id_vocab=global_node_id_vocab, delete_self_loop=delete_self_loop)
        self.edge_embedding = nn.Embedding(2, edge_size)  # [0 for internal context edges, 1 for interacting edges]

    @torch.no_grad()
    def construct_edges(self, pos_heavyatom, aa, atom_types, mask_atoms, block_lengths, lengths, fragment_type, structure_mask=None, sequence_mask=None):
        # Remove other atoms
        pos_heavyatom = pos_heavyatom[:, :, :self.max_num_atoms, :]
        atom_types = atom_types[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]
        if self.no_block:
            N = block_lengths.shape[0]
            block_lengths = torch.ones_like(atom_types, dtype=torch.int).reshape(N, -1)
            lengths = lengths*self.max_num_atoms
            fragment_type = torch.repeat_interleave(fragment_type, repeats=self.max_num_atoms, dim=1)
        else:
            block_lengths = torch.ones_like(block_lengths, dtype=torch.int)*self.max_num_atoms

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # (N, L)
        mask_residue_atom = mask_residue[:, :, None].expand(atom_types.shape)  # (N, L, max_num_atoms)

        if sequence_mask is not None:
            # Avoid data leakage at training time
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
            atom_types = torch.where(sequence_mask[:, :, None].expand(atom_types.shape), atom_types, torch.full_like(atom_types, fill_value=int(atomsymb_to_atomindex[atom_unk])))
        # if structure_mask is not None:
        #     # Avoid data leakage at training time
        #     pos_heavyatom = torch.where(structure_mask[:, :, None, None].expand(pos_heavyatom.shape), pos_heavyatom, torch.randn_like(pos_heavyatom) * 10)  # init_mask(pos_heavyatom, structure_mask)
        #     structure_mask = structure_mask[:, :, None].expand(atom_types.shape)

        pos_heavyatom = pos_heavyatom.reshape(-1, 1, 3)
        aa = aa.flatten()  # .detach().clone()
        atom_types = atom_types.flatten()
        mask_atoms = mask_atoms.flatten()
        block_lengths = block_lengths.flatten()
        lengths = lengths
        segment_id = fragment_type.flatten()
        segment_ids = torch.where(segment_id == 2, torch.tensor(1).to(segment_id.device), segment_id)
        mask_residue_atom = mask_residue_atom.flatten()

        mask = torch.logical_and(mask_residue_atom, mask_atoms)
        atoms_ids = torch.arange(0, len(atom_types)).to(mask.device)
        mask_atoms_id = atoms_ids[mask]
        # if structure_mask is not None:
        #     structure_mask = structure_mask.flatten()
        #     mask = torch.logical_and(mask, structure_mask)
        # generate_atoms_id = atoms_ids[mask]

        # batch_id and block_id
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            block_id = torch.zeros_like(atom_types)  # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

        S = aa
        if self.no_block:
            S = atom_types
        intra_edges, inter_edges, _, _, _ = self.edge_constructor(S, batch_id, segment_ids, X=pos_heavyatom,
                                                                  block_id=block_id, mask_atoms_id=mask_atoms_id,
                                                                #   generate_atoms_id=generate_atoms_id,
                                                                  )
        edges = torch.cat([intra_edges, inter_edges], dim=1)

        return block_id, batch_id, edges, intra_edges, inter_edges

    def forward(self, pos_heavyatom, aa, atom_types, mask_atoms, block_lengths, lengths, fragment_type, structure_mask=None, sequence_mask=None):

        block_id, batch_id, edges, intra_edges, inter_edges = self.construct_edges(pos_heavyatom, aa, atom_types, mask_atoms, block_lengths, lengths, fragment_type, structure_mask, sequence_mask)

        edge_attr = torch.cat([torch.zeros_like(intra_edges[0]), torch.ones_like(inter_edges[0])])
        edge_attr = self.edge_embedding(edge_attr)
        # if not self.no_block:
        #     edge_feat_pair = pair2edge(edges, lengths, pair_feat)
        #     edge_attr = edge_attr + edge_feat_pair

        return block_id, batch_id, edges, edge_attr

