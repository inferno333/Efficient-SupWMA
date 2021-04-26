
from __future__ import print_function
import argparse
import os
import random
import time
import h5py
import numpy as np
import pickle

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

# in order to import modules from pointnet folder
from utils.dataset import ORGDataset
from utils.model_supcon import PointNet_SupCon, PointNet_Classifier
from utils.logger import create_logger
from utils.metrics_plots import classify_report, per_class_metric, process_curves
from utils.metrics_plots import calculate_prec_recall_f1, best_swap, save_best_weights, gen_199_classify_report
from eval import kfold_evaluate_two_stage_contrastive_model
from utils.funcs import unify_path, makepath, fix_seed


# GPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    """load train and validation data"""
    # load feature and label data
    train_dataset = ORGDataset(
        root=args.input_path,
        logger=logger,
        num_fold=num_fold,
        k=args.k_fold,
        split='train')
    val_dataset = ORGDataset(
        root=args.input_path,
        logger=logger,
        num_fold=num_fold,
        k=args.k_fold,
        split='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True, num_workers=int(args.num_workers))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size,
        shuffle=False, num_workers=int(args.num_workers))

    train_data_size = len(train_dataset)
    val_data_size = len(val_dataset)
    logger.info('The training data size is:{}'.format(train_data_size))
    logger.info('The validation data size is:{}'.format(val_data_size))
    num_classes = len(train_dataset.label_names)
    logger.info('The number of classes is:{}'.format(num_classes))

    # load label names
    train_label_names = train_dataset.obtain_label_names()
    val_label_names = val_dataset.obtain_label_names()
    assert train_label_names == val_label_names
    label_names = train_label_names
    label_names_h5 = h5py.File(os.path.join(args.out_path, 'label_names.h5'), 'w')
    label_names_h5['y_names'] = label_names
    logger.info('The label names are: {}'.format(str(label_names)))

    return train_loader, val_loader, label_names, num_classes, train_data_size, val_data_size


def train_val_net(supcon_net, classify_net):
    """train and validation of the network"""
    time_start = time.time()