"""Reference from https://github.com/fxia22/pointnet.pytorch"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class PointNet_SupCon(nn.Module):
    """PointNet Encoder+Linear layers. Trained with contrastive loss"""
    def __init__(self, head='mlp', f