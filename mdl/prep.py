# --------------------------------------------------------------------------
# Prepare complex texts (Chemical Features) with rec, affinity, and lig (for MP complexes).
# --------------------------------------------------------------------------

import sys
import os.path as osp
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
from typing import Iterable

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
            doc_ofile=None, overwrite=True):
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


def run_steps(idx: int, rec_name: str, rec_doc: str, lig_name: str, lig_doc: str, config: MDLConfig):
    ''' Control overall flow using the functions of included steps above:
        - Pass docs (strings) to the next steps.
    '''

    rec_doc = rec_doc.strip()  # Remove empty first line if exists.
    rec = next(get_mol(rec_doc, doc_type='pdb'))

    lig = next(get_mol(lig_doc, doc_type='sdf'))

    # (1) Assign chemical feature (cf) to rec and pose doc.
    rec_cf = get_chemical_feature(rec, rec_name, None, 'rec')
    lig_cf = get_chemical_feature(lig, lig_name, None, 'lig')

    # (2) Generate doc including rec, affinity of complex, and lig.
    complex_name = f"{rec_name}+{lig_name}" if rec_name != lig_name else lig_name
    doc_odir = Path(f'{config.data_dir}').resolve()
    doc_odir.mkdir(parents=True, exist_ok=True)
    doc_ofile = f'{doc_odir}/{complex_name}.txt'
    get_doc(rec_cf, lig_cf, complex_name, None,
            doc_ofile, config.overwrite)


def prep(rec_file: str, lig_file: str, config: MDLConfig):
    ''' Generate rec-lig complex texts with pharmacophore nodes.
        Args:
        - pdb: string of multiple receptors
        - sdf: string of multiple ligands
        - config: MDLConfing
    '''

    # Check input files and generate docs.
    # pdb, sdf file should be the same name or pdb file contain only single element
    if not osp.isfile(rec_file):
        raise ValueError(f"File({rec_file}) not exists")
    if not osp.isfile(lig_file):
        raise ValueError(f"File({lig_file}) not exists")


    # Get text from ligand and receptor files
    # ligands
    lig_docs = [d + '\n$$$$\n' for d in [v for v in Path(lig_file).read_text().split('\n$$$$\n') if v.strip() != '']]
    lig_names = [osp.splitext(osp.basename(lig_file))[0]
                        + f'_{idx+1}' if len(lig_docs) > 1 else ''
                        for idx in range(len(lig_docs))]

    # receptors
    rec_doc = Path(rec_file).read_text()
    rec_docs = [
        d + 'END\n' for d in [v for v in rec_doc.split('END\n') if v.strip() != '']]
    rec_names = [osp.splitext(osp.basename(rec_file))[0]
                        + (f'_{idx+1}' if len(rec_docs) > 1 else '')
                        for idx in range(len(rec_docs))]
    
    if len(rec_docs) == len(lig_docs):
        pass
    elif len(rec_docs) != len(lig_docs) and len(rec_docs) == 1:
        # rec file contains only single structure
        rec_docs = rec_docs * len(lig_names)
        rec_names = rec_names * len(lig_names)
    else:
        raise ValueError("Receptor and lignad files are not paired properly.")

    # Run prep steps.
    num_processes = config.num_processes
    if num_processes > 1:
        with Pool(num_processes) as pool:
            pool.starmap(run_steps, zip(range(len(lig_docs)),
                                        rec_names, rec_docs, lig_names, lig_docs, repeat(config)))
    else:
        for idx, (rec_name, rec_doc, lig_name, lig_doc) in enumerate(tqdm(zip(rec_names, rec_docs, lig_names, lig_docs),
                                                                          total=len(lig_docs), desc='Prep complex texts')):
            run_steps(idx, rec_name, rec_doc, lig_name, lig_doc, config)


if __name__ == '__main__':
    pass
