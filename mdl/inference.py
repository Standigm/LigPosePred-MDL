# --------------------------------------------------------------------------
# Inference Module for Validation
# --------------------------------------------------------------------------

import sys
import os
import os.path as osp
import logging
import pandas as pd
from tqdm import tqdm
# from multiprocessing import Pool

import torch.multiprocessing as mp
import torch

from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, roc_curve, r2_score
)
from scipy.stats import pearsonr

from mdl.config import MDLConfig
from mdl.models.models import define_d
from mdl.datasets.infer_dataset import PosVoxelDataset
from mdl.infer_utils.trans3d import RotateVoxel
from mdl.infer_utils.loss import SigmoidFocalLoss
from mdl.infer_utils.metrics import reg_auc_score

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.float_format', '{:.3f}'.format)


class Validate():
    def __init__(self, config: MDLConfig):

        # Data params.
        self.task = config.task
        self.data_type = config.data_type
        self.node_feature_type = config.node_feature_type
        self.balance_ratio = config.balance_ratio
        self.label_corr = config.label_corr
        self.max_num_ps = config.max_num_ps
        self.label_th = config.label_th
        self.max_label = config.max_label
        self.transform = RotateVoxel() if config.rot else None

        # Model params.
        self.checkpoints = config.checkpoints  # checkpoints
        self.type_d = config.type_d
        self.node_feature_size = config.node_feature_size
        self.nkd = config.nkd
        self.out_dim = config.out_dim
        self.norm = config.norm
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.dropout = config.dropout
        self.out_dim = config.out_dim

        # Loss function params..
        self.use_focal_loss = config.use_focal_loss

        # Miscellenea
        self.batch_size = config.batch_size
        self.device = config.device

        # NOTE: CPU parallel inference is not efficient yet (slow).
        if str(self.device) == 'cpu':
            self.gpu_ids = [-1]
            self.num_processes = 1 
            if self.num_processes > 1:
                self.num_workers = 0
                torch.set_num_threads(1)
            else:
                self.num_workers = 8
        else:
            self.gpu_ids = [0]
            self.num_processes = 1
            self.num_workers = 8

        # Set loss funciton.
        self.loss_func = self.task_loss()

        # Load validation data.
        self.val_loader = self.load_data(config.data_dir)

    def val(self, checkpoints):
        num_processes = self.num_processes
        if num_processes > 1:
            with mp.Pool(num_processes) as p:
                names, scores = zip(*p.imap(
                    self.val_each, tqdm(checkpoints, desc='Infer with ensemble models')
                ))
            # Calling ray.get inside class with self arg are may be not supported.
            # ray.init(num_cpus=len(checkpoints))
            # names, scores = ray.get([self.val_each.remote(cp) for cp in checkpoints])
        else:
            names, scores = [], []
            for idx, cp in enumerate(tqdm(checkpoints, desc='Infer with  ensemble models'), 1):
                name, score = self.val_each(cp)
                names.append(name)
                scores.append(score)

        score_df = pd.DataFrame({'ligand': names[0]})
        for idx, score in enumerate(scores, 1):
            score_df.insert(idx, f'score_{idx}', score)

        score_df['score'] = score_df.mean(axis=1)
        mean_score_df = score_df[['ligand', 'score']]

        return score_df, mean_score_df

    # @ray.remote
    def val_each(self, checkpoint):
        model = self.load_model(checkpoint)
        model.eval()

        val_loader = self.val_loader
        num_iters = len(val_loader)
        loss_epoch = 0.
        pred_all, y_all, src_y_all, name_all = [], [], [], []

        # src_y: original target value (e.g., for regression)
        # y: classification label obtained from src_y by conf.label_th
        for batch in val_loader:
            image = batch[0].to(self.device)
            y, src_y = batch[1].to(self.device), batch[2].to(self.device)
            if self.out_dim == 1:
                y = y.unsqueeze(-1).float()
                src_y = src_y.unsqueeze(-1).float()
            name = batch[3]

            # Run model forward and compute loss
            pred = model(image).detach()
            loss = self.loss_func(pred, y)
            loss_epoch += loss

            if self.task == 'cls':
                pred = nn.Sigmoid()(pred) if self.out_dim == 1 else \
                       nn.Softmax(dim=1)(pred)
            pred_all.append(pred)
            y_all.append(y)
            src_y_all.append(src_y)
            name_all.append(name)

        loss_epoch = loss_epoch.item() / num_iters

        # Flatten lists for metrics.
        pred_all = torch.cat(pred_all).cpu()
        y_all = torch.cat(y_all).cpu()
        src_y_all = torch.cat(src_y_all).cpu()
        name_all = [y for x in name_all for y in x]

        score = pred_all.squeeze().tolist()
        # label = y_all.squeeze().tolist()
        # src_label = src_y_all.squeeze().tolist()

        return name_all, score

    def load_data(self, val_path):
        if self.data_type == 'aff':
            val_dataset = PosVoxelDataset(
                data_path=val_path,
                node_feature_type=self.node_feature_type,
                task=self.task,
                balance_ratio=self.balance_ratio,
                max_num_ps=self.max_num_ps,
                label_th=self.label_th,
                label_corr=self.label_corr,
                max_label=self.max_label,
                transform=self.transform
            )
        else:
            raise ValueError(f'Unkonwn data_type: {self.data_type}')

        if len(val_dataset) < 1:
            return None

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return val_loader

    def load_model(self, checkpoint):
        # Define model.
        # NOTE: node_feature_size not from datasets for loop over datasets.
        model = define_d(
            type_d=self.type_d,
            input_nc=self.node_feature_size,
            nkd=self.nkd,
            out_dim=self.out_dim,
            norm=self.norm,
            init_type='normal',
            init_gain=0.02,
            dropout=self.dropout,
            gpu_ids=self.gpu_ids,
        ).to(self.device)

        # Load checkpoint.
        logging.info(f'\n>>> Loading model {checkpoint}')

        checkpoint = torch.load(checkpoint, map_location=self.device)

        # remove module.* (DataParallel)
        checkpoint['model_state_dict'] = {k[7:] if k.startswith('module.') else k:v for k,v in checkpoint['model_state_dict'].items()}

        if str(self.device) == 'cpu':
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        # example_input = torch.rand(1, 18, 24, 24, 24) 
        # model = torch.jit.trace(model)

        return model

    def task_loss(self):
        ''' Define loss functions.'''
        if self.task == 'cls':  # for classification.
            if self.out_dim == 1:
                criterion = SigmoidFocalLoss() if self.use_focal_loss else \
                            nn.BCEWithLogitsLoss()
            else:
                criterion = nn.NLLLoss()
        elif self.task == 'reg':  # for regression.
            criterion = nn.L1Loss()
            # criterion = nn.MSELoss()
        else:
            raise ValueError('Select task type: cls|reg')

        return criterion.to(self.device)

    def get_metrics(self, bs_name, pred, y):
        # Classification metrics
        if self.task == 'cls':
            pred_y = (pred >= 0.5) if self.out_dim == 1 else torch.max(pred, dim=1)[1]
            acc = accuracy_score(y, pred_y)  # Accuracy
            ppv = precision_score(y, pred_y)  # Precision
            tpr = recall_score(y, pred_y)  # Recall
            auc = roc_auc_score(y, pred) if any(y < 1) and any(y > 0) else 0.0 # ROC-AUC
            df = pd.DataFrame([[bs_name, len(y), acc, ppv, tpr, auc]],
                              columns=['bs', 'n_poses', 'acc', 'ppv', 'tpr', 'auc'])
        # Regression metrics.
        elif self.task == 'reg':
            pr = pearsonr(y.reshape(-1), pred.reshape(-1))[0]  # Pearson correlation
            r2 = r2_score(y, pred)  # R^2
            # regression AUC
            auc = reg_auc_score(y, pred, num_rounds=3000)
            df = pd.DataFrame([[bs_name, len(y), pr, r2, auc]],
                              columns=['bs', 'n_poses', 'pr', 'r2', 'auc'])
        else:
            raise ValueError('Select task type: cls|reg')

        return df


def inference(config):
    # ## Infer and get scores over complex voxels..
    logging.info(f'\n>>> label_th {config.label_th}')
    logging.info(f'\n>>> Begin validation \n>> conf:\n{config}\n>>')

    val = Validate(config)

    # Validate each receptor for scoring.
    score_df, mean_score_df = val.val(config.checkpoints)

    logging.info(score_df)

    # Save and return scores.
    ofile = f'{osp.splitext(config.output_prefix)[0]}.csv'
    score_df.to_csv(ofile, index=False, float_format='%.3f')

    return mean_score_df


if __name__ == '__main__':
    pass
