# Basically for dataset ../input/phar2

import sys
import argparse
import bz2
import pickle
import _pickle as cPickle
import torch.multiprocessing as mp
import pandas as pd
import random
from glob import glob
from os import makedirs
from pathlib import Path
from typing import Union
from tqdm import tqdm
from time import time

import numpy as np
from torch.utils.data import Dataset, DataLoader

DTYPE = np.float32
DTYPE_INT = np.int32
DTYPE_STR = 'str_'
np.set_printoptions(threshold=np.inf)


class PosVoxelDataset(Dataset):
    ''' Voxel imgae (atom or pharmacophore) to label dataset.
    Input voxel feature
        Image: bs x nc x nv x nv x nv
        Ex) nc = 10 + 8, nv= 24
    Output feature
        Label: 0|1
    '''
    def __init__(
        self,
        data_path: str, node_feature_type: str = 'phar', task: str = 'cls',
        balance_ratio=1.0, max_num_ps: int = 30,
        label_th=[2.0, 4.0], label_corr='neg', max_label: float = -1.0,
        shuffle=False, transform=None, check=False
    ) -> None:
        ''' Create a dataset object
            Args:
            - node_feature_type: atom|phar
            - task: cls|reg
            - transform: random se3 tansformation
        '''
        # Feature properties.
        # NOTE: node_feature_size + 1 for type concatenation with one-hot.
        if node_feature_type == 'atom':
            # (number of node types + 1) * 2
            self.node_feature_size = 14
        elif node_feature_type == 'phar':
            # number of bs node types + 1 number of ps node types + 1
            self.node_feature_size = 18
        else:
            raise ValueError(f'Unknown node_feature_type: {node_feature_type}')

        self.node_feature_type = node_feature_type

        # Task: cls (classification) | reg (regression)
        self.task = task

        # Dataset and properties.
        assert Path(data_path).exists(), f'data_path not found: {data_path}'
        self.data_path = data_path
        self.balance_ratio = balance_ratio  # pos to neg ratio
        self.max_num_ps = max_num_ps
        self.th_lo = label_th[0]  # threshold of positve class.
        self.th_hi = label_th[1]  # threshold of negative class.
        self.label_corr = label_corr
        self.shuffle = shuffle  # random shuffle loaded data.

        # Set max label (depending on trainset distribution)
        self.max_label = max_label 

        # Miscellanea.
        self.check = check
        self.num_workers = 32  # 64

        # Load data.
        self.transform = transform
        self.encodings = self.load_data()
        self.len = len(self.encodings['labels'])
        print(f"Loaded dataset: {data_path} with length {self.len}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> dict:
        image, label, src_label, pid = tuple(
            [val[idx] for key, val in self.encodings.items()]
        )

        # Augmentation on the coordinates (e.g., random rotation).
        if self.transform:
            image = self.transform(image)

        return image, label, src_label, pid

        # NOTE: For possible conistency with Huggingface.
        # return {key: val[idx] for key, val in self.encodings.items()}

    def load_data(self) -> Union[dict, list]:
        ''' Read graphs and labels from all bs-ligand poses complex files.'''
        if Path(self.data_path).is_dir():
            data_files = glob(f'{self.data_path}/*.npz')
        else:
            data_files = pd.read_csv(self.data_path).values.tolist()
            data_files = [x[0] for x in data_files]
        data_files = sorted(data_files)
        # Define the list of images and labels of all complexes.
        images, labels, src_labels, pids = [], [], [], []
        if self.num_workers > 1:
            with mp.Pool(self.num_workers) as p:
                data = list(p.imap(
                    self.read_complex,
                    tqdm(data_files, desc='Pose voxel data')
                ))
            for bs_images, bs_labels, bs_src_labels, bs_pids in data:
                images += list(bs_images)
                labels += list(bs_labels)
                src_labels += list(bs_src_labels)
                pids += list(bs_pids)
        else:
            # NOTE: Label balancing done in each data_file.
            for data_file in tqdm(data_files, desc='Pose voxel data'):
                # Read complexes and flatten into single list.
                # NOTE: tag checkpoints --> checkpoint in return indicating flattened list.
                bs_images, bs_labels, bs_src_labels, bs_pids = self.read_complex(data_file)
                images += list(bs_images)
                labels += list(bs_labels)
                src_labels += list(bs_src_labels)
                pids += list(bs_pids)

            if self.shuffle:
                images, labels = zip(*random.shuffle(list(zip(images, labels))))

        return {'images': images, 'labels': labels, 'src_labels': src_labels, 'pids': pids}


    def read_complex(self, data_file) -> Union[dict, list]:
        '''
        Read imgages and labels from each bs-ligand poses complex (cp) file.
        '''
        # (1) Make pos(1) and neg(0) labels balanced if task == 'cls'.
        bs_images = np.asarray(np.load(data_file)['cp']).astype(DTYPE)
        bs_labels = np.asarray(np.load(data_file)['label']).astype(DTYPE)
        bs_pids = np.asarray([Path(data_file).stem]).astype(DTYPE_STR)

        if self.max_label > 0:
            idx_sampled = np.where(bs_labels <= self.max_label)[0]
            bs_images = bs_images[idx_sampled]
            bs_labels = bs_labels[idx_sampled]
            bs_pids = bs_pids[idx_sampled]

        if self.task == 'cls':
            if self.label_corr == 'neg':
                idx_pos = np.where(bs_labels <= self.th_lo)[0]
                idx_neg = np.where(bs_labels > self.th_hi)[0]
            elif self.label_corr == 'pos':
                idx_pos = np.where(bs_labels >= self.th_hi)[0]
                idx_neg = np.where(bs_labels < self.th_lo)[0]

            # Make number of poses per class <= max_num_ps // 2 and the same.
            if self.balance_ratio > 0:
                num_pos = len(idx_pos)
                num_neg = len(idx_neg)

                max_num_ps_class = self.max_num_ps // 2
                if (max_num_ps_class <= num_pos and
                    max_num_ps_class <= num_neg and
                    max_num_ps_class > 0):
                    idx_pos = np.random.choice(idx_pos, max_num_ps_class)
                    idx_neg = np.random.choice(idx_neg, max_num_ps_class)
                elif num_pos * self.balance_ratio <= num_neg:
                    idx_neg = np.random.choice(idx_neg, int(num_pos * self.balance_ratio))
                elif num_neg * self.balance_ratio <= num_pos:
                    idx_pos = np.random.choice(idx_pos, int(num_neg * self.balance_ratio))

            # NOTE: to return original values for validation.
            bs_src_labels = np.copy(bs_labels)

            bs_labels[idx_pos] = 1
            bs_labels[idx_neg] = 0

            idx = np.arange(len(bs_labels))
            idx_sampled = np.sort(np.concatenate((idx_pos, idx_neg)))
            bs_labels[np.setdiff1d(idx, idx_sampled)] = -1

            # NOTE: comment out for data without labels are controlled in val script.
            # bs_images = bs_images[idx_sampled]
            # bs_labels = bs_labels[idx_sampled]

        # Select max_num_ps poses of the lowest label values.
        elif self.task == 'reg':
            if self.max_num_ps > 0:
                idx_sampled = np.argsort(bs_labels)
                bs_images = bs_images[idx_sampled][:self.max_num_ps]
                bs_labels = bs_labels[idx_sampled][:self.max_num_ps]
                bs_pids = bs_pids[idx_sampled][:self.max_num_ps]

        return bs_images, bs_labels, bs_src_labels, bs_pids


if __name__ == '__main__':
    pass
