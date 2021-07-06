import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    def __init__(self):
        super(PointNetfeat, self).__init__()
        # 3-layer MLP (via 1D-CNN) : encoder points individually
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv