import os
import sys
import types

from pathlib import Path
from mdl.prep_utils.common import get_doc

try:
    from openeye import oechem
except ImportError:
    class OENotInstalled:
        def __getattribute__(self, __name: str):
            raise ImportError("Openeye is not installed")
    oechem = OENotInstalled()

class Atom():
    def __init__(self, atom_name=None, res_name=None, chain_id=None, res_seq=-100, res_seq_nr=-100, r=[0.0, 0.0, 0.0]):
        self.atom_name = atom_name
        self.res_name = res_name
        self.chain_id = chain_id
        self.res_seq = res_seq
        self.res_seq_nr = res_seq_nr
        self.r = r


def write_residue_in_pdb_format(atoms):
    doc_sub = ''
    for atom in atoms:
        doc_sub += (
            f"{'ATOM': <12}{atom.atom_name} {atom.res_name} "
            f"{atom.chain_id}{atom.res_seq}{' ': <4}"
            f"{atom.r[0]:8.3f}{atom.r[1]:8.3f}{atom.r[2]:8.3f}\n"
        )
    return doc_sub


def write_chemical_feature(res_name, atoms):
    # HD: hydrogen bond donor (side chain)
    # HA: hydrogen bond acceptor (side chain)
    # PC: positively charged
    # NC: negatively charged
    # RG: ring
    # HP: hydrophobe
    # AD: either acceptor or donor
    # HDM: hydrogen bond donor (main chain)
    # HAM: hydrogen bond acceptor (main chain)
    # NA: not assigned

    doc_sub = ''
    cfs = []
    name = ''

    if res_name == 'ARG':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'NE' or name == 'NH1' or name == 'NH2':
                cf = Atom(atom_name=' PC ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'CB' or name == 'CG' or name == 'CD':
                cf = Atom(atom_name=' HP ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'HIS':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'ND1' or name == 'NE2':
                cf = Atom(atom_name=' PC ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'CG' or name == 'CD2' or name == 'CE1':
                cf = Atom(atom_name=' RG ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'LYS':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'NZ':
                 cf = Atom(atom_name=' PC ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                 cfs.append(cf)
            elif name == 'CB' or name == 'CG' or name == 'CD':
                cf = Atom(atom_name=' HP ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'ASP':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'OD1' or name == 'OD2':
                cf = Atom(atom_name=' NC ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'GLU':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'OE1' or name == 'OE2':
                cf = Atom(atom_name=' NC ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'SER':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'OG':
                cf = Atom(atom_name=' AD ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'THR':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'OG1':
                cf = Atom(atom_name=' AD ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'CG2':
                cf = Atom(atom_name=' HP ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'ASN':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'ND2':
                cf = Atom(atom_name=' HD ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'OD1':
                cf = Atom(atom_name=' HA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'GLN':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'NE2':
                cf = Atom(atom_name=' HD ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'OE1':
                cf = Atom(atom_name=' HA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'CYS':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'SG':
                cf = Atom(atom_name=' HP ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'GLY':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'PRO':
        for atom in atoms:
            name = atom.atom_name
            if name == 'O' or name == 'N':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'CA' or name == 'CB' or name == 'CG' or name == 'CD':
                cf = Atom(atom_name=' RG ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'ALA':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'CB':
                cf = Atom(atom_name=' HP ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'VAL':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'CB' or name == 'CG1' or name == 'CG2':
                cf = Atom(atom_name=' HP ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'ILE':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'CB' or name == 'CG1' or name == 'CG2' or name == 'CD1':
                cf = Atom(atom_name=' HP ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'LEU':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'CG' or name == 'CD1' or name == 'CD2':
                cf = Atom(atom_name=' HP ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'MET':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'CE':
                cf = Atom(atom_name=' HP ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'PHE':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'CG' or name == 'CD1' or name == 'CD2' or name == 'CE1' or name == 'CE2' or name == 'CZ':
                cf = Atom(atom_name=' RG ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'TYR':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'OH':
                cf = Atom(atom_name=' AD ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'CG' or name == 'CD1' or name == 'CD2' or name == 'CE1' or name == 'CE2' or name == 'CZ':
                cf = Atom(atom_name=' RG ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)
    elif res_name == 'TRP':
        for atom in atoms:
            name = atom.atom_name
            if name == 'N':
                cf = Atom(atom_name=' HDM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'NE1':
                cf = Atom(atom_name=' HD ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'O':
                cf = Atom(atom_name=' HAM', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            elif name == 'CE3' or name == 'CZ2' or name == 'CZ3' or name == 'CH2' or name == 'CG' or name == 'CD1' or name == 'CD2' or name == 'CE2':
                cf = Atom(atom_name=' RG ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
            else:
                cf = Atom(atom_name=' NA ', res_name=atom.res_name, chain_id=atom.chain_id, res_seq=atom.res_seq, res_seq_nr='', r=[atom.r[0], atom.r[1], atom.r[2]])
                cfs.append(cf)
        doc_sub = write_residue_in_pdb_format(cfs)

    return doc_sub


def assign_rec_chemical_feature(mol_doc, outpdb=None, doc_type='pdb'):
    prev_resseq = -1000
    resseq = 0
    prev_resname = ''
    resname = ''

    atoms = []
    if isinstance(mol_doc, oechem.OEGraphMol):
        mol = get_doc(mol_doc, doc_type)
        lines = mol.splitlines()
    elif Path(mol_doc).is_file():
        mol = Path(mol_doc).read_text().strip().split('END\n')
        lines = mol.splitlines()

    doc = ''
    for line in lines:
        if line.startswith('TER'):
            doc += write_chemical_feature(prev_resname, atoms)
            doc += 'TER\n'
            break

        if line.startswith('ATOM') and line[12:16].strip()[0] != 'H':
            resseq = int(line[22:26].strip())
            resname = line[17:20]

            if (resseq != prev_resseq) and (len(atoms) != 0):
                doc += write_chemical_feature(prev_resname, atoms)
                atoms.clear()

            r = [float(line[30:38].strip()),
                 float(line[38:46].strip()),
                 float(line[46:54].strip())]
            atom = Atom(
                atom_name=line[12:16].strip(), res_name=resname,
                chain_id=line[21:22], res_seq=line[22:26],
                res_seq_nr=line[22:27].strip(), r=r
            )
            atoms.append(atom)

            prev_resseq = resseq
            prev_resname = resname

    doc += 'END\n'

    if outpdb:
        Path(outpdb).write_text(doc)

    return doc


if __name__ == "__main__":
    pass
