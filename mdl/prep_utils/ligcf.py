import os
import math
import types
import numpy as np
from pathlib import Path

try:
    from openeye import oechem
    from openeye import oeshape
except ImportError:
    class OENotInstalled:
        def __getattribute__(self, __name: str):
            raise ImportError("Openeye is not installed")
    oechem = OENotInstalled()
    oeshape = OENotInstalled()


from mdl.prep_utils.common import get_mol, get_doc


dict_cf_name = {}
dict_cf_name[0] = 'NA'   # not assigned
dict_cf_name[1] = 'HD'   # hydrogen bond donor (defined using a single atom)
dict_cf_name[2] = 'HA'   # hydrogen bond acceptor (defined using a single atom)
dict_cf_name[3] = 'PC'   # cation (only for protein)
dict_cf_name[4] = 'NC'   # anion (only for protein)
dict_cf_name[5] = 'RG'   # ring
dict_cf_name[6] = 'HP'   # hydrophobe
dict_cf_name[7] = 'AD'   # either acceptor or donor

dict_name_cf = {}
dict_name_cf['NA'] = 0
dict_name_cf['HD'] = 1
dict_name_cf['HA'] = 2
dict_name_cf['PC'] = 3
dict_name_cf['NC'] = 4
dict_name_cf['RG'] = 5
dict_name_cf['HP'] = 6
dict_name_cf['AD'] = 7


class CFUnit:
    def __init__(self):
        self.cf_type = 0
        self.r = [0.0, 0.0, 0.0]


def get_distance(r1, r2):
    return math.sqrt((r1[0]-r2[0])**2 + (r1[1]-r2[1])**2 + (r1[2]-r2[2])**2)


def set_hydroxyl_groups(list_cf):
    for i in range(len(list_cf)):
        for j in range(i+1, len(list_cf), 1):
            if ((list_cf[i].cf_type == 1 and list_cf[j].cf_type == 2) or
                    (list_cf[i].cf_type == 2 and list_cf[j].cf_type == 1)):
                if get_distance(list_cf[i].r, list_cf[j].r) == 0:
                    list_cf[i].cf_type = 7
                    list_cf[j].cf_type = -1

    for elem in list(list_cf):
        if elem.cf_type == -1:
            list_cf.remove(elem)


def get_num_hydroxyl(list_cf):
    num_hydroxyl = 0
    for elem in list(list_cf):
        if elem.cf_type == 7:
            num_hydroxyl += 1

    return num_hydroxyl


def get_cf_lig_mol(lig_mol, cff):
    num_atoms = lig_mol.NumAtoms()
    np_atom_coords = np.zeros((num_atoms, 3), dtype=float)
    np_atom_cf = np.zeros((num_atoms,), dtype=int)

    for atom in lig_mol.GetAtoms():
        atom_idx = atom.GetIdx()
        coords = oechem.OEFloatArray(3)
        lig_mol.GetCoords(atom, coords)
        np_atom_coords[atom_idx][0] = coords[0]
        np_atom_coords[atom_idx][1] = coords[1]
        np_atom_coords[atom_idx][2] = coords[2]

    prep = oeshape.OEOverlapPrep()
    prep.Prep(lig_mol)

    cf = []
    for atom in oeshape.OEGetColorAtoms(lig_mol):
        cf_type = cff.GetType(atom.GetName())
        if cf_type == 5 or cf_type == 6:
            p_atoms = oeshape.OEGetColorParents(atom)
            for p_atom in p_atoms:
                p_atom_idx = p_atom.GetIdx()
                np_atom_cf[p_atom_idx] = cf_type

    for atom in oeshape.OEGetColorAtoms(lig_mol):
        cf_type = cff.GetType(atom.GetName())
        if cf_type == 1 or cf_type == 2:
            p_atoms = oeshape.OEGetColorParents(atom)
            for p_atom in p_atoms:
                p_atom_idx = p_atom.GetIdx()
                if (np_atom_cf[p_atom_idx] == 1 and cf_type == 2) or (np_atom_cf[p_atom_idx] == 2 and cf_type == 1):
                    np_atom_cf[p_atom_idx] = 7
                else:
                    np_atom_cf[p_atom_idx] = cf_type

    for i in range(num_atoms):
        cfunit = CFUnit()
        cfunit.cf_type = np_atom_cf[i]
        cfunit.r[0] = np_atom_coords[i][0]
        cfunit.r[1] = np_atom_coords[i][1]
        cfunit.r[2] = np_atom_coords[i][2]
        cf.append(cfunit)

    return cf


def assign_lig_chemical_feature(mol_doc, outpdb=None, doc_type='sdf'):
    cff = oeshape.OEColorForceField()
    if not cff.Init(oeshape.OEColorFFType_ImplicitMillsDean):
        oechem.OEThrow.Error("Unable to inititialize color forcefield")

    if isinstance(mol_doc, oechem.OEGraphMol):
        mol = mol_doc
    elif Path(mol_doc).is_file():
        mol = get_mol(mol_doc)

    doc = ''
    cf_mol = get_cf_lig_mol(mol, cff)
    for idx, elem in enumerate(cf_mol, 1):
        cf_name = dict_cf_name[elem.cf_type]
        doc += (
            f"{'ATOM': <13}{cf_name: <4}LIG A{idx:4}{' ': <4}"
            f"{elem.r[0]:8.3f}{elem.r[1]:8.3f}{elem.r[2]:8.3f}\n"
        )
    doc += 'TER\nEND\n'

    if outpdb:
        Path(outpdb).write_text(doc)

    return doc


if __name__ == "__main__":
    pass
