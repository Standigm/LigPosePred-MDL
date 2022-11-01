# #######################################################
# MDL (SE3CNN) model inference
#   - Version 1.0
#   - 2022. 8. 28.
#   - Woojoo Sim, Woong-Gi Chang
# #######################################################

import sys
import os
import os.path as osp
import logging
import pandas as pd
import numpy as np
import torch
import argparse
import random
import tempfile
import glob

from pathlib import Path
from time import time

from mdl.config import MDLConfig
from mdl.prep import prep
from mdl.voxel_prep import voxel_prep
from mdl.inference import inference

from typing import Union, Iterable

import warnings
warnings.filterwarnings("ignore")


pd.set_option('display.float_format', '{:.3f}'.format)


def init_logger(log_file: Union[None, str] = None):
    from imp import reload
    reload(logging)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        # handlers=[logging.FileHandler(log_file, 'w+'), logging.StreamHandler(),
        handlers=[logging.FileHandler(log_file, 'w+')]
    )


def set_random_seed(seed, deterministic=False):
    """Set manual random seed seed for python random/numpy/torch/torch.cuda module.
    Args:
        seed (int): Random Seed to use.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prep_complex_text(pdb_file: str,
                      sdf_file: str,
                      config: MDLConfig):
    ''' Generate complex texts. complexts contains coordinates with voxel channel index for each point.
        format: x, y, z, channel_idx
    ...
    Args:
        - pdb_file: receptor pdb
        - sdf_file: sdf including multiple ligands
        - config:  MDLConfig
    '''
    # #################
    # (1) Prep data.
    # #################

    # ## (1-1) Prep pharmacophore complex texts.
    # text contains x,y,z,channel_index
    logging.info(f'\n>>> Generate complex texts in {data_dir}:\n')
    prep(pdb_file, sdf_file, config)
    print(f'\n>>> Complex texts generated in {data_dir}')


def prep_voxel(binding_site: Iterable, config: MDLConfig):
    ''' Convert complex texts to voxel image in npz format 
    Args:
        - config:  MDLConfig
    '''
    # ## (1-2) Prep pharmacophore 3D voxel tensors (3D images).
    logging.info(f'\n>>> Generate complex voxels in {data_dir}:\n')
    voxel_prep(binding_site, config)
    print(f'\n>>> Complex voxels generated in {data_dir}')


def predict_pose(config: MDLConfig):
    ''' Predict Pose with confidence score (0.0-1.0)
    Args:
        - config:  MDLConfig
    '''

    # #################
    # (2) Inference
    # #################

    logging.info('\n\n>>> Infer pIC50 scores:\n')
    print(f"\n>>> Inference on {config.device}...")
    mean_score_df = inference(config)
    print(
        f"\n>>> MDL (SE3CNN) rescore calculated at {f'{osp.splitext(config.output_prefix)[0]}.csv'}")

    return mean_score_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MDL Pose Prediction")
    parser.add_argument('-p', '--pdb_file', type=str, required=True,
                        help='input receptor pdb file')
    parser.add_argument('-l', '--sdf_file', type=str, required=True,
                        help='input ligand sdf file')
    parser.add_argument('-o', '--output_prefix', type=str,
                        default=None, help='output prefix')
    parser.add_argument('-po', '--prep_odir', type=str, default=None,
                        help='pre-processed data directory where voxel inputs are stored. '
                        'if not set, temporal directory was used.')
    parser.add_argument('-np', '--num_processes', type=int,
                        default=1, help='The number of process for parallel data prep execution')
    parser.add_argument('-bs', '--binding_site', type=str, default=None,
                        help='use pre-calculated binding site coordinates. '
                        'if not provided, ligand center will be used. ex) 53.0,32.1,32.4')
    parser.add_argument('--use_gpu', action='store_true',
                        help='use gpu for inference')
    args = parser.parse_args()
    # binding site
    if args.binding_site is not None:
        args.binding_site = tuple([float(v)
                                  for v in args.binding_site.split(',')])
        if len(args.binding_site) != 3:
            raise ValueError(f"Invalid binding_site. {args.binding_site}")

    # Check input files.
    if not Path(args.pdb_file).is_file():
        raise ValueError(f'{args.pdb_file}: File not found.')
    if not Path(args.sdf_file).is_file():
        raise ValueError(f'{args.sdf_file}: File not found.')

    # data_dir
    if args.prep_odir:
        os.makedirs(args.prep_odir, exist_ok=True)
        data_dir = args.prep_odir
    else:
        temp_dir = tempfile.TemporaryDirectory()
        data_dir = temp_dir.name

    # set default output_prefix as sdf_file name
    if args.output_prefix is None:
        args.output_prefix = osp.splitext(args.sdf_file)[0]

    # Load and set config file.
    config = MDLConfig(
        data_dir=data_dir,
        output_prefix=args.output_prefix,
        num_processes=args.num_processes,
        device='cuda:0' if args.use_gpu else 'cpu')

    # Fix seed for random numbers.
    if not config.seed:
        config.seed = 1992  # np.random.randint(100000)
        set_random_seed(config.seed)

    
    # Define save path.
    Path(osp.dirname(config.output_prefix)).mkdir(parents=True, exist_ok=True)
    

    # Set logging.
    log_file = f'{config.output_prefix}.log'
    init_logger(log_file)
    print(f'\n>>> Start MDL (SE3CNN) pose prediction:'
          f'\n>>> Logs are written in {log_file}.'
          f'\n>>> Input pdb: {args.pdb_file}.'
          f'\n>>> Input sdf: {args.sdf_file}.')

    start = time()
    # prep data
    if len(glob.glob(f"{config.data_dir}/*.txt")) == 0:
        prep_complex_text(args.pdb_file, args.sdf_file, config)
    if len(glob.glob(f"{config.data_dir}/*.npz")) == 0 or \
            (len(glob.glob(f"{config.data_dir}/*.npz")) != len(glob.glob(f"{config.data_dir}/*.txt"))):
        prep_voxel(args.binding_site, config)

    # inference
    mean_score_df = predict_pose(config)
    # print(mean_score_df)
    print(f'Elapsed: {time() - start:.2f}s')
