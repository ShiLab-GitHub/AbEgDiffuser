import torch
from Bio.PDB import Selection
from Bio.PDB.Residue import Residue
from easydict import EasyDict
from Bio import PDB

from .constants import (
    AA, max_num_heavyatoms,
    restype_to_heavyatom_names, 
    BBHeavyAtom,
    atom_pad, atomsymb_to_atomindex,
    atom_pos_pad, possymb_to_posindex,
)


class ParsingException(Exception):
    pass

def _get_pos_code(atom_name, element):
    if atom_name == '':
        pos_code = possymb_to_posindex(atom_pos_pad)
    else:
        pos_code = atom_name.lstrip(element)
        pos_code = ''.join((c for c in pos_code if not c.isdigit()))
    return pos_code

def _get_residue_heavyatom_info(res: Residue):
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)
    atom_types = torch.full_like(mask_heavyatom, atomsymb_to_atomindex[atom_pad], dtype=torch.long)
    atom_positions = torch.full_like(mask_heavyatom, possymb_to_posindex[atom_pos_pad], dtype=torch.long)
    restype = AA(res.get_resname())
    for idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
        if atom_name == '': continue
        if atom_name in res:
            pos_heavyatom[idx] = torch.tensor(res[atom_name].get_coord().tolist(), dtype=pos_heavyatom.dtype)
            mask_heavyatom[idx] = True
            atom_types[idx] = torch.tensor(atomsymb_to_atomindex[res[atom_name].element])
            atom_positions[idx] = torch.tensor(possymb_to_posindex[_get_pos_code(atom_name, res[atom_name].element)])
    return pos_heavyatom, mask_heavyatom, atom_types, atom_positions


def parse_biopython_structure(entity, unknown_threshold=1.0, max_resseq=None):
    chains = Selection.unfold_entities(entity, 'C')
    chains.sort(key=lambda c: c.get_id())
    data = EasyDict({
        'chain_id': [],
        'resseq': [], 'icode': [], 'res_nb': [],
        'aa': [],
        'pos_heavyatom': [], 'mask_heavyatom': [],
        'atom_types': [], 'atom_positions': []
    })
    tensor_types = {
        'resseq': torch.LongTensor,
        'res_nb': torch.LongTensor,
        'aa': torch.LongTensor,
        'pos_heavyatom': torch.stack,
        'mask_heavyatom': torch.stack,
        'atom_types': torch.stack,
        'atom_positions': torch.stack,
    }

    count_aa, count_unk = 0, 0

    for i, chain in enumerate(chains):
        seq_this = 0   # Renumbering residues
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))   # Sort residues by resseq-icode
        for _, res in enumerate(residues):
            resseq_this = int(res.get_id()[1])
            if max_resseq is not None and resseq_this > max_resseq:
                continue

            resname = res.get_resname()
            if not AA.is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue
            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK: 
                count_unk += 1
                continue

            # Chain info
            data.chain_id.append(chain.get_id())

            # Residue types
            data.aa.append(restype) # Will be automatically cast to torch.long

            # Heavy atoms
            pos_heavyatom, mask_heavyatom, atom_types, atom_positions = _get_residue_heavyatom_info(res)
            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)
            data.atom_types.append(atom_types)
            data.atom_positions.append(atom_positions)

            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]
            if seq_this == 0:
                seq_this = 1
            else:
                d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_CA_CA <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)

            data.resseq.append(resseq_this)
            data.icode.append(icode_this)
            data.res_nb.append(seq_this)

    if len(data.aa) == 0:
        raise ParsingException('No parsed residues.')

    if (count_unk / count_aa) >= unknown_threshold:
        raise ParsingException(
            f'Too many unknown residues, threshold {unknown_threshold:.2f}.'
        )

    seq_map = {}
    for i, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode)):
        seq_map[(chain_id, resseq, icode)] = i

    for key, convert_fn in tensor_types.items():
        data[key] = convert_fn(data[key])

    return data, seq_map

if __name__ == '__main__':
    parser = PDB.PDBParser(QUIET=True)
    id = '7DK2'
    pdb_path = './data/examples/7DK2_AB_C.pdb'
    model = parser.get_structure(id, pdb_path)[0]

    data, seq_map = parse_biopython_structure(
        model,
        max_resseq=106  # Chothia, end of Light chain Fv
    )
    print(f"data.aa = {data.aa}")
    print(f"data.aa.shape = {data.aa.shape}")
    print(f"data.atom_types[0] = {data.atom_types[0]}")
    print(f"data.atom_types.shape = {data.atom_types.shape}")
    print(f"data.atom_positions[0] = {data.atom_positions[0]}")
    print(f"data.atom_positions.shape = {data.atom_positions.shape}")

