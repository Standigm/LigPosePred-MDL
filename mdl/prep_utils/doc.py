import sys
import argparse
import numpy as np
from pathlib import Path

dict_cf_name = {}
dict_cf_name[0] = 'NA'   # not assigned
dict_cf_name[1] = 'HD'   # hydrogen bond donor (side chain)
dict_cf_name[2] = 'HA'   # hydrogen bond acceptor (side chain)
dict_cf_name[3] = 'PC'   # cation
dict_cf_name[4] = 'NC'   # anion
dict_cf_name[5] = 'RG'   # ring
dict_cf_name[6] = 'HP'   # hydrophobe
dict_cf_name[7] = 'AD'   # either acceptor or donor
dict_cf_name[8] = 'HDM'  # hydrogen bond donor (main chain)
dict_cf_name[9] = 'HAM'  # hydrogen bond acceptor (main chain)

dict_name_cf = {}
dict_name_cf['NA'] = 0
dict_name_cf['HD'] = 1
dict_name_cf['HA'] = 2
dict_name_cf['PC'] = 3
dict_name_cf['NC'] = 4
dict_name_cf['RG'] = 5
dict_name_cf['HP'] = 6
dict_name_cf['AD'] = 7
dict_name_cf['HDM'] = 8
dict_name_cf['HAM'] = 9


class Atom():
    def __init__(self, r=[0.0, 0.0, 0.0], cf=0):
        self.r = r
        self.cf = cf


class Molecule():
    def __init__(self):
        self.atoms = []

    def add_atom(self, atom):
        self.atoms.append(atom)


def read_mol(mol_doc, is_file=False):
    ''' Read file or string in PDB format.'''
    mol = Molecule()
    if is_file:
        lines = Path(mol_doc).read_text().strip().splitlines()
    else:
        lines = mol_doc.strip().split('\n')

    for line in lines:
        if line.startswith('ATOM'):
            array = line.split()
            if len(array) < 8:
                pc_type = array[1]
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atom = Atom([x, y, z], dict_name_cf.get(pc_type))
                mol.add_atom(atom)
            else:
                pc_type = array[1]
                x = float(array[-3])
                y = float(array[-2])
                z = float(array[-1])
                atom = Atom([x, y, z], dict_name_cf.get(pc_type))
                mol.add_atom(atom)

    return mol


def write_mol(doc, fo):
    mol = read_mol(doc)
    for atom in mol.atoms:
        fo.write(f'{atom.r[0]:.3f},{atom.r[1]:.3f},{atom.r[2]:.3f},{atom.cf}\n')
    fo.write('\n')


# NOTE: For affinity prediction, term label means affinity here.
def gen_doc_fixed(rec_doc, lig_doc, lig_name,
                  lig_label_file=None, ofile=None, is_file=False):

    # NOTE: strip() needed to exclude empty lines for split.
    if lig_doc:
        if is_file:
            lig_docs = Path(lig_doc).read_text().strip().split('END\n')
        else:
            lig_docs = lig_doc.strip().split('END\n')
        n_ligs = len(lig_docs)

        if lig_label_file:
            lig_labels = np.loadtxt(lig_label_file, delimiter=',').reshape(-1,)
        else:
            lig_labels = -100.0 * np.ones(n_ligs)
        n_lig_labels = len(lig_labels)

        # Check n ligs (poses) and n labels are the same to each other.
        if n_ligs != n_lig_labels:
            print(f'{lig_name} : Error checking num consistancy of ligs and labels.')
            return

    fo = open(ofile, 'w')

    rec = read_mol(rec_doc, is_file)
    for atom in rec.atoms:
        fo.write(f'{atom.r[0]:.3f},{atom.r[1]:.3f},{atom.r[2]:.3f},{atom.cf}\n')
    fo.write('\n')

    if lig_doc:
        for doc, label in zip(lig_docs, lig_labels):
            fo.write(f'{label:.3f}\n\n')
            write_mol(doc, fo)

    return doc


def main(args):
    rec_file = args.rec_file
    pos_file = args.pos_file
    label_file = args.label_file

    name = Path(rec_file).stem.replace('_rec', '')
    pid = f'{Path(rec_file).parent.parent.stem}'
    traj = f'{Path(rec_file).parent.stem}'
    odir = Path(f'{args.odir}/{pid}/{traj}')
    odir.mkdir(parents=True, exist_ok=True)
    ofile = f'{odir}/{name}_doc.txt'

    gen_doc_fixed(rec_file, ofile, pos_file, label_file)


DSET = 'check'
REC = f'../../input/ss/{DSET}_pos_aligned/1a30/md_1/1a30_11_rec_cf.pdb'
POS = f'../../input/ss/{DSET}_pos_aligned/1a30/md_1/1a30_11_pos_cf.pdb'
RMSD = f'../../input/ss/{DSET}_pos_aligned/1a30/md_1/1a30_11_pos_label.csv'
ODIR = f'../../input/ss/{DSET}_doc/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rec_file', type=str, default=REC)
    parser.add_argument('--pos_file', type=str, default=POS)
    parser.add_argument('--label_file', type=str, default=RMSD)
    parser.add_argument('--odir', type=str, default=ODIR)
    args = parser.parse_args()

    main(args)

