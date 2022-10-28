import argparse
import os
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from tqdm import tqdm

try:
    from openeye import oechem
except ImportError:
    class OENotInstalled:
        def __getattribute__(self, __name: str):
            raise ImportError("Openeye is not installed")
    oechem = OENotInstalled()


def get_mol(mol_in, doc_type=None, out_doc_type=None):
    ''' Get mol or mol doc (string) from mol file or string.'''
    ims = oechem.oemolistream()

    if doc_type is not None:
        # Load mol from string.
        # Identify string format.
        if doc_type == 'sdf':
            ims.SetFormat(oechem.OEFormat_SDF)
        elif doc_type == 'pdb':
            ims.SetFormat(oechem.OEFormat_PDB)
        else:
            raise ValueError(
                f"doc_type {doc_type} not recognized: use sdf|pdb.")
        ims.openstring(mol_in)
    else:
        # Load mol From file
        # File format identified by suffix.
        ims.open(mol_in)

    # Generator of multiple mols.
    mols = ims.GetOEGraphMols()

    # Return mol doc (string) instead of mol generator.
    if out_doc_type is not None:
        oms = oechem.oemolostream()

        mol_docs = []
        for mol in mols:
            oms = oechem.oemolostream()

            if out_doc_type == 'sdf':
                oms.SetFormat(oechem.OEFormat_SDF)
            elif out_doc_type == 'pdb':
                oms.SetFormat(oechem.OEFormat_PDB)
            else:
                raise ValueError(
                    f"out_doc_type {out_doc_type} not recognized: use sdf|pdb.")

            oms.openstring()
            oechem.OEWriteMolecule(oms, mol)
            mol_doc = oms.GetString().decode('UTF-8')
            mol_docs.append(mol_doc)

        return mol_docs

    else:
        return mols


def get_doc(mol, doc_type):
    ''' Get mol doc (string) from mol generator.'''
    oms = oechem.oemolostream()

    if doc_type == 'sdf':
        oms.SetFormat(oechem.OEFormat_SDF)
    elif doc_type == 'pdb':
        oms.SetFormat(oechem.OEFormat_PDB)
    else:
        raise ValueError(
            'Mol file type not recognized: use sdf|pdb.')

    oms.openstring()
    oechem.OEWriteMolecule(oms, mol)
    mol_doc = oms.GetString().decode('UTF-8')

    return mol_doc


def split_mol(mol_file, odir, ostem_by_title=True, ostem=None, doc_type=None):
    mols = get_mol(mol_file, doc_type)
    ostems = []
    for idx, mol in enumerate(tqdm(mols), 1):
        if ostem_by_title:
            title_elements = lig.GetTitle().split('|')
            ostem = f"{title_elements[0]}_{title_elements[-1]}"
            ostems.append(ostem)
            ofile = f'{odir}/{ostem}_lig.sdf'
        elif ostem:
            ofile = f'{odir}/{ostem}_{idx}.sdf'
        else:
            ofile = f'{odir}/{Path(mol_file).stem}_{idx}.sdf'

        oms = oechem.oemolostream()
        oms.open(ofile)
        oechem.OEWriteMolecule(oms, mol)

    return ostems, idx


def merge_mols(mol_files, ofile):
    ims = oechem.oemolistream()
    oms = oechem.oemolostream()

    oms.open(ofile)
    for mol_file in mol_files:
        ims.open(mol_file)
        for mol in ims.GetOEGraphMols():
            oechem.OEWriteMolecule(oms, mol)


def get_coords(mol_file, doc_type=None):
    ''' Get coords from each mol (all including atoms).'''
    mols = get_mol(mol_file, doc_type)
    coords_mols = []
    for mol in mols:
        num_atoms = mol.GetMaxAtomIdx()
        coords_mol = oechem.OEFloatArray(3 * num_atoms)
        mol.GetCoords(coords_mol)
        coords_mols.append(coords_mol)

    return np.reshape(coords_mols, (len(coords_mols), 3, -1))


def get_coords_exc_h(mol_file, doc_type=None):
    ''' Get coords from each atom, excluding Hs.'''
    mols = get_mol(mol_file, doc_type)
    coords_mols = []
    for mol in mols:
        coords_mol = []
        for atom in mol.GetAtoms():
            if atom.IsHydrogen():
                continue
            coords = oechem.OEFloatArray(3)
            mol.GetCoords(atom, coords)
            coords_mol.append(coords)
        coords_mols.append(coords_mol)

    return np.asarray(coords_mols)


# - include a residue if any node in the residue located < r.
def crop_pdb(pdb_file, ref_mol_file, r_int, keep_residue=False):
    ref_xyzs = get_coords(ref_mol_file)
    lines = Path(pdb_file).read_text().splitlines(True)
    atoms = [x for x in lines if x.startswith(
        'ATOM') and x[12:16].strip()[0] != 'H']

    doc = []
    if keep_residue:
        res_atoms = []
        for idx, atom in enumerate(atoms):
            res = atom[17:20]
            if idx < 1:
                pres = res
            if res == pres:
                res_atoms.append(atom)
                pres = res
                continue
            for atom in res_atoms:
                xyz = [float(atom[30:38].strip()),
                       float(atom[38:46].strip()),
                       float(atom[46:54].strip())]
                dist = cdist(xyz, ref_xyzs)
                if dist.min() <= r_int:
                    doc += atom
            res_atoms = []
    else:
        for atom in atoms:
            xyz = [float(atom[30:38].strip()),
                   float(atom[38:46].strip()),
                   float(atom[46:54].strip())]
            dist = cdist(xyz, ref_xyzs)
            if dist.min() <= r_int:
                doc += atom
    doc += 'TER\n'

    return doc


MOL = '../../input/mp/mp103_raw/mp103_lig_mer.sdf'
ODIR = '../../input/mp/mp103_aff/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mol_file', type=str, default=MOL)
    parser.add_argument('--odir', type=str, default=ODIR)
    args = parser.parse_args()

    mol_file = args.mol_file

    n_mols = split_mol(args.mol_file, args.odir)
    print(n_mols)
