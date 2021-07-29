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
            raise ValueError('Head not supported: {}. Please select from "mlp" or "linear"'.format(head))

    def forward(self, x):
        global_feat = self.encoder(x)
        # contrastive feature
        contra_feat = F.normalize(self.head(global_feat), dim=1)  # normalization is important

        return contra_feat


class PointNetfeat(nn.Module):
    def __init__(self):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=Tru