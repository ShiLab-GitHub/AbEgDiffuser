#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

def extract_false_ranges(structure_mask):
    structure_mask = structure_mask.flatten()
    start_indices = []
    end_indices = []
    in_false_segment = False
    start = 0

    for i in range(structure_mask.size(0)):
        if not structure_mask[i] and not in_false_segment:
            in_false_segment = True
            start = i
        elif structure_mask[i] and in_false_segment:
            in_false_segment = False
            start_indices.append(start)
            end_indices.append(i - 1)

    ranges = [(start, end) for start, end in zip(start_indices, end_indices)]
    return ranges

def init_mask(coord, structure_mask):
    generate_ranges = extract_false_ranges(structure_mask)
    N, L, n_channel, n_dim = coord.shape
    mask_coord = coord.clone().reshape(N * L, n_channel, n_dim)
    # n_channel, n_dim = coord.shape[1:]
    for start, end in generate_ranges:
        l_coord, r_coord = mask_coord[start - 1], mask_coord[end + 1]  # [n_channel, 3]
        n_span = end - start + 2
        coord_offsets = (r_coord - l_coord)[None, :, :].expand(n_span - 1, n_channel, n_dim)  # [n_mask, n_channel, 3]
        coord_offsets = torch.cumsum(coord_offsets, dim=0)
        mask_coords = l_coord + coord_offsets / n_span
        mask_coord[start:end + 1] = mask_coords
    return mask_coord.reshape(N, L, -1, 3)

def stable_norm(input, *args, **kwargs):
    return torch.norm(input, *args, **kwargs)
    input = input.clone()
    with torch.no_grad():
        sign = torch.sign(input)
        input = torch.abs(input)
        input.clamp_(min=1e-10)
        input = sign * input
    return torch.norm(input, *args, **kwargs)


def _unit_edges_from_block_edges(unit_block_id, block_src_dst, Z=None, k=None):
    '''
    :param unit_block_id [N], id of block of each unit. Assume X is sorted so that block_id starts from 0 to Nb - 1
    :param block_src_dst: [Eb, 2], all edges (block level), represented in (src, dst)
    '''
    block_n_units = scatter_sum(torch.ones_like(unit_block_id), unit_block_id)  # [Nb], number of units in each block
    block_offsets = F.pad(torch.cumsum(block_n_units[:-1], dim=0), (1, 0), value=0)  # [Nb]
    edge_n_units = block_n_units[block_src_dst]  # [Eb, 2], number of units at two end of the block edges
    edge_n_pairs = edge_n_units[:, 0] * edge_n_units[:, 1]  # [Eb], number of unit-pairs in each edge

    # block edge id for unit pairs
    edge_id = torch.zeros(edge_n_pairs.sum(), dtype=torch.long,
                          device=edge_n_pairs.device)  # [Eu], which edge each unit pair belongs to
    edge_start_index = torch.cumsum(edge_n_pairs, dim=0)[
                       :-1]  # [Eb - 1], start index of each edge (without the first edge as it starts with 0) in unit_src_dst
    edge_id[edge_start_index] = 1
    edge_id = torch.cumsum(edge_id, dim=0)  # [Eu], which edge each unit pair belongs to, start from 0, end with Eb - 1

    # get unit-pair src-dst indexes
    unit_src_dst = torch.ones_like(edge_id)  # [Eu]
    unit_src_dst[edge_start_index] = -(edge_n_pairs[
                                       :-1] - 1)  # [Eu], e.g. [1,1,1,-2,1,1,1,1,-4,1], corresponding to edge id [0,0,0,1,1,1,1,1,2,2]
    del edge_start_index  # release memory
    if len(unit_src_dst) > 0:
        unit_src_dst[0] = 0  # [Eu], e.g. [0,1,1,-2,1,1,1,1,-4,1], corresponding to edge id [0,0,0,1,1,1,1,1,2,2]
    unit_src_dst = torch.cumsum(unit_src_dst,
                                dim=0)  # [Eu], e.g. [0,1,2,0,1,2,3,4,0,1], corresponding to edge id [0,0,0,1,1,1,1,1,2,2]
    unit_dst_n = edge_n_units[:, 1][
        edge_id]  # [Eu], each block edge has m*n unit pairs, here n refers to the number of units in the dst block
    # turn 1D indexes to 2D indexes (TODO: this block is memory-intensive)
    unit_src = torch.div(unit_src_dst, unit_dst_n, rounding_mode='floor') + block_offsets[
        block_src_dst[:, 0][edge_id]]  # [Eu]
    unit_dst = torch.remainder(unit_src_dst,
                               unit_dst_n)  # [Eu], e.g. [0,1,2,0,0,0,0,0,0,1] for block-pair shape 1*3, 5*1, 1*2
    unit_dist_local = unit_dst
    # release some memory
    del unit_dst_n, unit_src_dst  # release memory
    unit_edge_src_start = (unit_dst == 0)
    unit_dst = unit_dst + block_offsets[block_src_dst[:, 1][edge_id]]  # [Eu]
    del block_offsets, block_src_dst  # release memory
    unit_edge_src_id = unit_edge_src_start.long()
    if len(unit_edge_src_id) > 1:
        unit_edge_src_id[0] = 0
    unit_edge_src_id = torch.cumsum(unit_edge_src_id, dim=0)  # [Eu], e.g. [0,0,0,1,2,3,4,5,6,6] for the above example

    if k is None:
        return (unit_src, unit_dst), (edge_id, unit_edge_src_start, unit_edge_src_id)

    # sparsify, each atom is connected to the nearest k atoms in the other block in the same block edge

    D = torch.norm(Z[unit_src] - Z[unit_dst], dim=-1)  # [Eu, n_channel]
    D = D.sum(dim=-1)  # [Eu]

    max_n = torch.max(scatter_sum(torch.ones_like(unit_edge_src_id), unit_edge_src_id))
    k = min(k, max_n)

    BIGINT = 1e10  # assign a large distance to invalid edges
    N = unit_edge_src_id.max() + 1
    # src_dst = src_dst.transpose(0, 1)  # [2, Ef]
    dist = torch.norm(Z[unit_src] - Z[unit_dst], dim=-1).sum(-1)  # [Eu]

    dist_mat = torch.ones(N, max_n, device=dist.device, dtype=dist.dtype) * BIGINT  # [N, max_n]
    dist_mat[(unit_edge_src_id, unit_dist_local)] = dist
    del dist
    dist_neighbors, dst = torch.topk(dist_mat, k, dim=-1, largest=False)  # [N, topk]
    del dist_mat

    src = torch.arange(0, N, device=dst.device).unsqueeze(-1).repeat(1, k)
    unit_edge_src_start = torch.zeros_like(src).bool()  # [N, k]
    unit_edge_src_start[:, 0] = True
    src, dst = src.flatten(), dst.flatten()
    unit_edge_src_start = unit_edge_src_start.flatten()
    dist_neighbors = dist_neighbors.flatten()
    is_valid = dist_neighbors < BIGINT
    src = src.masked_select(is_valid)
    dst = dst.masked_select(is_valid)
    unit_edge_src_start = unit_edge_src_start.masked_select(is_valid)

    # extract row, col and edge id
    mat = torch.ones(N, max_n, device=unit_src.device, dtype=unit_src.dtype) * -1
    mat[(unit_edge_src_id, unit_dist_local)] = unit_src
    unit_src = mat[(src, dst)]
    mat[(unit_edge_src_id, unit_dist_local)] = unit_dst
    unit_dst = mat[(src, dst)]
    mat[(unit_edge_src_id, unit_dist_local)] = edge_id
    edge_id = mat[(src, dst)]

    unit_edge_src_id = src

    return (unit_src, unit_dst), (edge_id, unit_edge_src_start, unit_edge_src_id)

def _knn_edges(dist, src_dst, k_neighbors, batch_info):
    '''
    :param dist: [Ef], given distance of edges
    :param src_dst: [Ef, 2], full possible edges represented in (src, dst)
    '''
    offsets, batch_id, max_n, gni2lni = batch_info

    k_neighbors = min(max_n, k_neighbors)

    BIGINT = 1e10  # assign a large distance to invalid edges
    N = batch_id.shape[0]
    src_dst = src_dst.transpose(0, 1)  # [2, Ef]

    dist_mat = torch.ones(N, max_n, device=dist.device, dtype=dist.dtype) * BIGINT  # [N, max_n]
    dist_mat[(src_dst[0], gni2lni[src_dst[1]])] = dist
    del dist
    dist_neighbors, dst = torch.topk(dist_mat, k_neighbors, dim=-1, largest=False)  # [N, topk]

    src = torch.arange(0, N, device=dst.device).unsqueeze(-1).repeat(1, k_neighbors)
    src, dst = src.flatten(), dst.flatten()
    dist_neighbors = dist_neighbors.flatten()
    is_valid = dist_neighbors < BIGINT
    src = src.masked_select(is_valid)
    dst = dst.masked_select(is_valid)

    dst = dst + offsets[batch_id[src]]  # mapping from local to global node index

    edges = torch.stack([src, dst])  # message passed from dst to src

    return edges  # [2, E]

def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res

def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


class BatchEdgeConstructor:
    '''
    Construct intra-segment edges (intra_edges) and inter-segment edges (inter_edges) with O(Nn) complexity,
    where n is the largest number of nodes of one graph in the batch.
    Additionally consider global nodes:
        global nodes will connect to all nodes in its segment (global_normal_edges)
        global nodes will connect to each other regardless of the segments they are in (global_global_edges)
    Additionally consider edges between adjacent nodes in the sequence in the same segment (seq_edges)
    '''

    def __init__(self, global_node_id_vocab=[], delete_self_loop=True) -> None:
        self.global_node_id_vocab = copy(global_node_id_vocab)
        self.delete_self_loop = delete_self_loop

        # buffer
        self._reset_buffer()

    def _reset_buffer(self):
        self.row = None
        self.col = None
        self.row_global = None
        self.col_global = None
        self.row_seg = None
        self.col_seg = None
        self.offsets = None
        self.max_n = None
        self.gni2lni = None
        self.not_global_edges = None
        # torch.cuda.empty_cache()

    def get_batch_edges(self, batch_id):

        # construct tensors to map between global / local node index
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs]
        # global node index to local index. lni2gni can be implemented as lni + offsets[batch_id]
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]  # [N]

        # all possible edges (within the same graph)
        # same bid (get rid of self-loop and none edges)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        # shift right and pad 1 to the left
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        if self.delete_self_loop:
            same_bid[(gni, gni2lni)] = 0  # delete self loop
        row, col = torch.nonzero(same_bid).T  # [2, n_edge_all]
        col = col + offsets[batch_id[row]]  # mapping from local to global node index
        return (row, col), (offsets, max_n, gni2lni)

    def _prepare(self, S, batch_id, segment_ids) -> None:
        (row, col), (offsets, max_n, gni2lni) = self.get_batch_edges(batch_id)
        # print(f"row = {row}")  # zyb 测试
        # print(f"col = {col}")  # zyb 测试

        # not global edges
        if len(self.global_node_id_vocab):
            is_global = sequential_or(*[S == global_node_id for global_node_id in self.global_node_id_vocab])  # [N]
        else:
            is_global = torch.zeros_like(S, dtype=torch.bool)
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))

        # segment ids
        row_seg, col_seg = segment_ids[row], segment_ids[col]
        # print(f"segment_ids = {segment_ids}")  # zyb 测试
        # print(f"row_seg = {row_seg}")  # zyb 测试
        # print(f"col_seg = {col_seg}")  # zyb 测试

        # add to buffer
        self.row, self.col = row, col
        self.offsets, self.max_n, self.gni2lni = offsets, max_n, gni2lni
        self.row_global, self.col_global = row_global, col_global
        self.not_global_edges = not_global_edges
        self.row_seg, self.col_seg = row_seg, col_seg

    def _construct_intra_edges(self, S, batch_id, segment_ids, **kwargs):
        row, col = self.row, self.col
        # all possible ctx edges: same seg, not global
        # print(f"self.row_seg == self.col_seg = {self.row_seg == self.col_seg}")  # zyb 测试
        select_edges = torch.logical_and(self.row_seg == self.col_seg, self.not_global_edges)
        # print(f"select_edges = {select_edges}")  # zyb 测试
        intra_all_row, intra_all_col = row[select_edges], col[select_edges]
        return torch.stack([intra_all_row, intra_all_col])

    def _construct_inter_edges(self, S, batch_id, segment_ids, **kwargs):
        row, col = self.row, self.col
        # all possible inter edges: not same seg, not global
        # print(f"self.row_seg != self.col_seg = {self.row_seg != self.col_seg}")  # zyb 测试
        select_edges = torch.logical_and(self.row_seg != self.col_seg, self.not_global_edges)
        # print(f"select_edges = {select_edges}")  # zyb 测试
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        return torch.stack([inter_all_row, inter_all_col])

    def _construct_global_edges(self, S, batch_id, segment_ids, **kwargs):
        row, col = self.row, self.col
        # edges between global and normal nodes
        select_edges = torch.logical_and(self.row_seg == self.col_seg, torch.logical_not(self.not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        select_edges = torch.logical_and(self.row_global, self.col_global)  # self-loop has been deleted
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return global_normal, global_global

    def _construct_seq_edges(self, S, batch_id, segment_ids, **kwargs):
        row, col = self.row, self.col
        # add additional edge to neighbors in 1D sequence (except epitope)
        select_edges = sequential_and(
            torch.logical_or((row - col) == 1, (row - col) == -1),  # adjacent in the graph order
            self.not_global_edges  # not global edges (also ensure the edges are in the same segment)
            # self.row_seg != self.ag_seg_id  # not epitope
        )
        seq_adj = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return seq_adj

    @torch.no_grad()
    def __call__(self, S, batch_id, segment_ids, **kwargs):
        '''
        Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch
        '''
        # prepare inputs
        self._prepare(S, batch_id, segment_ids)

        # intra-segment edges
        intra_edges = self._construct_intra_edges(S, batch_id, segment_ids, **kwargs)

        # inter-segment edges
        inter_edges = self._construct_inter_edges(S, batch_id, segment_ids, **kwargs)

        # edges between global nodes and normal/global nodes
        global_normal_edges, global_global_edges = self._construct_global_edges(S, batch_id, segment_ids, **kwargs)

        # edges on the 1D sequence
        seq_edges = self._construct_seq_edges(S, batch_id, segment_ids, **kwargs)

        self._reset_buffer()

        return intra_edges, inter_edges, global_normal_edges, global_global_edges, seq_edges




