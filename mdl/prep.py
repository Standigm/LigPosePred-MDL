# --------------------------------------------------------------------------
# Prepare complex texts (Chemical Features) with rec, affinity, and lig (for MP complexes).
# --------------------------------------------------------------------------

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm
from shutil import copy
from math import log10

from mdl.config import MDLConfig
from mdl.prep_utils.common import get_mol
from mdl.prep_utils.reccf import assign_rec_chemical_feature
from mdl.prep_utils.ligcf import assign_lig_chemical_feature
from mdl.prep_utils.doc import gen_doc_fixed

# Set pandas and numpy printing formats.
pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', '{:.3f}'.format)
# pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(precision=3, threshold=np.inf)

def chem_feat_func(mol_type):
    if mol_type == 'rec':
        chem_feat_fn = assign_rec_chemical_feature
    elif mol_type == 'lig':
        chem_feat_fn = assign_lig_chemical_feature
    else:
        raise ValueError("Invalid mol_type:{mol_type}")
    return chem_feat_fn

def get_chemical_feature(mol_file, mol_name, mol_ofile, mol_type, overwrite=False):
    if mol_ofile:
        if Path(mol_ofile).exists() and not overwrite:
            print(f'{mol_ofile} already exists!')
            return
    try:
        mol_cf = chem_feat_func(mol_type)(mol_file, mol_ofile)
    except:
        print(f'{mol_name} : Error assigning chemical features.')
        return
    logging.info(f'{mol_name} : chemical features assigned successfully.')

    return mol_cf


def get_doc(rec_doc, lig_doc, lig_name, lig_label_file=None,
            doc_ofile=None, doc_type='fixed', overwrite=True):
    if doc_ofile:
        if Path(doc_ofile).exists() and not overwrite:
            print(f'{doc_ofile} already exists!')
            return
    try:
        gen_doc_fixed(
            rec_doc, lig_doc, lig_name, lig_label_file, doc_ofile, is_file=False
        )
    except:
        print(f'{doc_ofile} : Error generating doc.')
        return
    logging.info(f'{doc_ofile} : doc generated successfully.')


def run_steps(lig_doc: str, rec_cf, config: MDLConfig):
    ''' Control overall flow using the functions of included steps above:
        - Pass docs (strings) to the next steps.
        - Params:
            lig: oechem.OEGraphMol
    '''
    lig_doc = lig_doc.strip()  # Remove empty first line if exists.
    lig = next(get_mol(lig_doc, doc_type='sdf'))
    # title_elements = lig.GetTitle().split('|')
    # lig_name = f"{title_elements[0]}_{title_elements[-1]}"
    lig_name = lig.GetTitle()

    # (1) Assign chemical feature (cf) to rec and pose doc.
    lig_cf = get_chemical_feature(lig, lig_name, None, 'lig')

    # (2) Generate doc including rec, affinity of complex, and lig. 
    doc_odir = Path(f'{config.data_dir}').resolve()
    doc_odir.mkdir(parents=True, exist_ok=True)
    doc_ofile = f'{doc_odir}/{lig_name}.txt'
    get_doc(rec_cf, lig_cf, lig_name, None,
            doc_ofile, config.doc_type, config.overwrite)


def prep(pdb_file: str, sdf_file: str, config: MDLConfig):
    ''' Generate rec-lig complex texts with pharmacophore nodes.
        Args:
        - pdb_file: receptor pdb
        - sdf_file: sdf including multiple ligands
        - config: MDLConfing
    '''
    # Load ligands as docs (strings) instead of OEGraphMol for starmap.
    ligs = [d+'$$$$\n' for d in Path(sdf_file).read_text().split('$$$$\n')[:-1]]

    # Load receptor as OEGraphMol and convert to cf txt.
    rec = next(get_mol(pdb_file))
    rec_name = Path(pdb_file).stem
    rec_cf = get_chemical_feature(rec, rec_name, None, 'rec', config.rec_suffix)

    # Run prep steps.
    num_processes = config.num_processes
    if num_processes > 1:
        pool = Pool(num_processes) if num_processes else Pool()
        pool.starmap(run_steps, zip(ligs, repeat(rec_cf), repeat(config)))
        pool.close()
    else:
        for lig in tqdm(ligs, desc='Prep complex texts'):
            run_steps(lig, rec_cf, config)


if __name__ == '__main__':
    pass

