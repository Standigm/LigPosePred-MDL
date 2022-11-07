import sys
import pandas as pd
import numpy as np
import random
import logging
from io import StringIO
from os import makedirs
from pathlib import Path
from glob import glob
from re import split
from multiprocessing import Pool
from itertools import repeat
from scipy.spatial.distance import cdist
from tqdm import tqdm
from typing import Iterable
from mdl.config import MDLConfig

from mdl.prep_utils.voxel import (
    crop_grid,
    gen_pfield,
    gen_dfield,
    save_dfield
)

pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
np.set_printoptions(precision=3, threshold=np.inf, linewidth=160)

DTYPE = 'float16'
DTYPE_INT = 'int32'


def read_features(nodes):
    ''' Read nodes to dataframes. '''
    # Define columns for node dataframes
    node_columns = ['x', 'y', 'z', 'node_type']

    # Read nodes.
    df_nodes = pd.read_csv(StringIO(nodes), names=node_columns)
    # Delete rows with node_type 0.
    # NOTE: indices of df maintained to crop nodes and dist later.
    df_nodes = df_nodes.drop(df_nodes[df_nodes['node_type'] == 0].index)

    return df_nodes


def crop_nodes(df_nodes, df_nodes_ref, config):
    ''' Crop bs nodes by min and max of xyz of ligand ps. '''
    r_int = config.r_int

    idx_cropped = []
    if config.crop_type == 'node':
        xyzs_ref = df_nodes_ref.iloc[:, :3].to_numpy()
        for idx, row in zip(df_nodes.index.to_numpy(), df_nodes.to_numpy()):
            xyz = [row[:3]]
            dist = cdist(xyz, xyzs_ref)
            if dist.min() <= r_int:
                idx_cropped.append(idx)
    elif config.crop_type == 'box':
        v_min, v_max = df_nodes_ref.min(axis=0), df_nodes_ref.max(axis=0)
        x_min, x_max = v_min['x'] - r_int, v_max['x'] + r_int
        y_min, y_max = v_min['y'] - r_int, v_max['y'] + r_int
        z_min, z_max = v_min['z'] - r_int, v_max['z'] + r_int
        idx_cropped += df_nodes.index[
            (df_nodes['x'] > x_min) & (df_nodes['x'] < x_max) &
            (df_nodes['y'] > y_min) & (df_nodes['y'] < y_max) &
            (df_nodes['z'] > z_min) & (df_nodes['z'] < z_max)].tolist()
    else:
        raise ValueError('Select crop_type: node|box')

    return df_nodes.loc[idx_cropped, :]


def center_nodes_(df_nodes_bs, df_nodes_ps, center=None):
    if center is None:
        center = df_nodes_ps[['x', 'y', 'z']].mean().values
    df_nodes_bs[['x', 'y', 'z']] -= center
    df_nodes_ps[['x', 'y', 'z']] -= center


def check_df(dfs):
    for idx, df in enumerate(dfs):
        if df.empty:
            return idx
    return -1


def load_complex(data_file, config):
    ''' Read mols from each binding site (bs) - ligand ps complex file.
        Structure of each complex file (docs):
            (docs are separated by empty lines)
            doc: bs1 xyzs and node types
            doc: bs1-ps1 label
            doc: ps1 xyzs and node types
            ...
        NOTE: Graph features for complex: positive for bs, negative for lig.
    '''
    # Split complex text into bs and ligand ps.
    raw_doc = split(r'\n\t?\n', data_file.read_text().strip())

    # Read bs feature dataframes 
    raw_nodes_bs = raw_doc[0]
    df_nodes_bs = read_features(raw_nodes_bs)

    # Read bs and ps feature dataframes (nodes).
    num_rows = 2 if config.node_feature_type == 'phar' else 3  # rows for each pose
    idx_label = range(num_rows - 1, len(raw_doc), num_rows)
    labels, nodes_ps = [], []  # all ps for the same bs
    for idx in idx_label:
        label = float(raw_doc[idx])
        labels.append(label)
        raw_nodes_psx = raw_doc[idx+1]
        df_nodes_psx = read_features(raw_nodes_psx)
        nodes_ps.append(df_nodes_psx)

    if nodes_ps:
        df_nodes_ps = pd.concat(nodes_ps, keys=list(range(len(idx_label))))

    else:
        df_nodes_ps = pd.DataFrame(columns=df_nodes_bs.columns)


    return df_nodes_bs, df_nodes_ps, labels


def gen_voxel(df_nodes_bs, df_nodes_ps, labels,
              num_ps, cname, odir, out_file, config):
    # (5) Generate density fields (dfields) and save each of them.
    # NOTE: Intermediately generate point fields (pfields).
    ch_type = config.node_feature_type
    nv = config.nv
    vs = config.vs
    deff = config.deff
    dfield_type = config.dfield_type
    norm_type = config.dfield_norm_type

    # dfield complex pose, label complex pose
    dfield_cp, label_cp = [], []
    for idx in range(num_ps):
        df_nodes_bsx = df_nodes_bs.loc[idx].reset_index(drop=True)
        df_nodes_psx = df_nodes_ps.loc[idx].reset_index(drop=True)

        # Genrerate pfields.
        pfield_bsx = gen_pfield(
            df_nodes_bsx, 'bs', cname, ch_type, nv, vs
        )
        pfield_psx = gen_pfield(
            df_nodes_psx, 'ps', cname, ch_type, nv, vs
        )

        # Genrerate dfields.
        dfield_bsx = gen_dfield(
            pfield_bsx, nv, deff, dfield_type, norm_type, ch_type
        )
        dfield_psx = gen_dfield(
            pfield_psx, nv, deff, dfield_type, norm_type, ch_type
        )
        dfield_cpx = np.concatenate((dfield_bsx, dfield_psx))
        dfield_cpx = dfield_cpx.astype(DTYPE)
        label_cpx = np.asarray(labels[idx]).astype(DTYPE)

        # Save density fields.
        if config.save_voxel == 'mer':
            dfield_cp.append(dfield_cpx)
            label_cp.append(label_cpx)
        elif config.save_voxel == 'sep':
            features = {'cp': dfield_cpx, 'label': label_cpx}
            out_file = f'{odir}/{cname}_p{idx}.npz'
            save_dfield(features, out_file)

    if config.save_voxel == 'mer':
        features = {'cp': dfield_cp, 'label': label_cp}
        save_dfield(features, out_file)


def run_steps(data_file, binding_site, df_nodes_ps_set, config):
    # Define output file and skip if exists (with not conf.overwrite)
    data_file = Path(data_file)
    cname = data_file.stem
    odir = config.data_dir
    if binding_site is not None:
        if not isinstance(binding_site, np.ndarray):
            binding_site = np.array(binding_site)

    if config.save_voxel == 'mer':
        out_file = f'{odir}/{cname}.npz'
    elif config.save_voxel == 'sep':
        out_file = f'{odir}/{cname}_p0.npz'
    if not config.overwrite and Path(out_file).exists():
        print(f'{out_file} already exists!')
        return

    # (1) Load complex dict (of dataframes):
    #   - df_node_bs: dataframe of nodes of bs (binding site)
    #   - df_node_ps: dataframe of nodes of ps (multiple poses)
    #   - labels: list of labels or rms values of ps
    try:
        df_nodes_bs, df_nodes_ps, labels = load_complex(data_file, config)
    except:
        logging.info(f'>>> {data_file} >> Error: reading node features.')
        return
    num_ps = len(labels)

    # (2) Crop bs nodes interacting (distance < r_int) with ligand ps.
    # * Drop complexes of small bs or small psx.
    th_bs, th_ps = 4, 2
    if config.r_int > 0:
        try:
            df_nodes_bs = crop_nodes(df_nodes_bs, df_nodes_ps_set, config)
        except:
            logging.info(f'>>> {data_file} >> Error: cropping nodes.')
            return
    if len(df_nodes_bs) <= th_bs or len(df_nodes_ps.loc[0]) <= th_ps:
        logging.info(
            f'[{cname}] Error: small binding site or poses after crop.')
        return

    # (3) Center and crop the coordinates of nodes.
    # Ceter coords.
    if config.norm_type == 'sep':
        nodes_bs, nodes_ps = [], []
        for idx in range(num_ps):
            df_nodes_bsx = df_nodes_bs.copy()
            df_nodes_psx = df_nodes_ps.loc[idx]
            center_nodes_(df_nodes_bsx, df_nodes_psx, center=binding_site)
            nodes_bs.append(df_nodes_bsx)
            nodes_ps.append(df_nodes_psx)
        df_nodes_bs = pd.concat(nodes_bs, keys=list(range(num_ps)))
        df_nodes_ps = pd.concat(nodes_ps, keys=list(range(num_ps)))
    elif config.norm_type == 'mer':
        center_nodes_(df_nodes_bs, df_nodes_ps, center=binding_site)
        nodes_bs = []
        for idx in range(num_ps):
            nodes_bs.append(df_nodes_bsx)
        df_nodes_bs = pd.concat(nodes_bs, keys=list(range(num_ps)))

    # Crop grid to conf.nv.
    nv = config.nv
    vs = config.vs
    df_nodes_bs = crop_grid(df_nodes_bs, cname, nv, vs)
    df_nodes_ps = crop_grid(df_nodes_ps, cname, nv, vs)

    # Cast data types.
    df_nodes_bs = df_nodes_bs.astype(DTYPE)
    df_nodes_ps = df_nodes_ps.astype(DTYPE)
    df_nodes_bs = df_nodes_bs.astype({'node_type': DTYPE_INT})
    df_nodes_ps = df_nodes_ps.astype({'node_type': DTYPE_INT})

    # (4) Check empty df.
    idx_empty = check_df([df_nodes_bs, df_nodes_ps])
    if idx_empty > -1:
        logging.info(f'[{cname}] Error: empty df ({idx_empty}).')
        return

    # (5) Generate density fields (dfields) and save each of them.
    # NOTE: Intermediately generate point fields (pfields).
    try:
        gen_voxel(df_nodes_bs, df_nodes_ps, labels,
                  num_ps, cname, odir, out_file, config)
    except:
        print(f'>>> {data_file} >> Error: generating fields.')
        return


def get_ps_set(data_files, config):
    nodes_ps = []
    for data_file in data_files:
        raw_doc = split(r'\n\t?\n', Path(data_file).read_text().strip())
        raw_nodes_psx = raw_doc[2]
        df_nodes_psx = read_features(raw_nodes_psx)
        nodes_ps.append(df_nodes_psx)

    df_nodes_ps = pd.concat(nodes_ps, keys=list(range(len(nodes_ps))))

    return df_nodes_ps


def voxel_prep(binding_site: Iterable, config: MDLConfig):
    ''' Convert bs-lig_pose complex files:
        (1) Select interacting bs points around ligands'
        (2) Normalize point xyz
    '''
    # Run prep.
    data_files = sorted(glob(f'{config.data_dir}/*.txt'))

    if config.crop_mode == 'all':
        # NOTE: Get the set of all ligand poses in the input set.
        # This is to crop the receptor refering to the set of all poses.
        df_nodes_ps_sets = repeat(get_ps_set(data_files, config))
    else:
        # seperate crop
        df_nodes_ps_sets = [get_ps_set([data_file], config)
                            for data_file in data_files]

    num_processes = config.num_processes
    if num_processes > 1:
        with Pool(num_processes) as pool:
            pool.starmap(run_steps, zip(data_files, repeat(
                binding_site), df_nodes_ps_sets, repeat(config)))
    else:
        for data_file, df_nodes_ps_set in tqdm(zip(data_files, df_nodes_ps_sets),
                                               total=len(data_files), desc='Prep voxels'):
            run_steps(data_file, binding_site, df_nodes_ps_set, config)


if __name__ == '__main__':
    pass
