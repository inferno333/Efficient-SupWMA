"""Reference from https://github.com/fxia22/pointnet.pytorch"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class PointNet_SupCon(nn.Module):
    """PointNet Encoder+Linear layers. Trained with contrastive loss"""
    def __init__(self, head='mlp', feat_dim=128):
        super(PointNet_SupCon, self).__init__()
        # encoder
        self.encoder = PointNetfeat()
        # Contrastive learning
        if head == 'linear':
            self.head = nn.Linear(1024, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, feat_dim)
            )
        else:
            