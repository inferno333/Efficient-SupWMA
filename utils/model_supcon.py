"""Reference from https://github.com/fxia22/pointnet.pytorch"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class 