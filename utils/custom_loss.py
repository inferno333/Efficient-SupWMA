# Reference from https://github.com/HobbitLong/SupContrast Yonglong Tian (yonglong@mit.edu)
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive lo