
from utils.logger import create_logger

import whitematteranalysis as wma
import numpy as np

import argparse
import h5py
import time
import os
import pickle

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from utils.model import PointNetCls
from utils.model_supcon import PointNet_SupCon, PointNet_Classifier
from utils.dataset import TestDataset
from utils.metrics_plots import classify_report


def load_test_data():
    """Load test data and labels name in model"""
    # Put test data into loader
    test_dataset = TestDataset(args.feat_path, args.input_label_path, args.label_names,