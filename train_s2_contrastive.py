
from __future__ import print_function
import argparse
import os
import random
import copy
import time
import h5py
import numpy as np
import pickle

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data


from utils.dataset import SupConDataset
from utils.model_supcon import PointNet_SupCon
from utils.logger import create_logger
from utils.custom_loss import SupConLoss
from utils.funcs import fix_seed, unify_path, makepath

# GPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    """load train and validation data"""
    # load feature and label data
    train_dataset = SupConDataset(
        root=args.input_path,
        logger=logger,
        num_fold=num_fold,
        k=args.k_fold,
        split='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True, num_workers=int(args.num_workers))

    train_data_size = len(train_dataset)
    logger.info('The training data size is:{}'.format(train_data_size))
    num_classes = len(train_dataset.label_names)
    logger.info('The number of classes is:{}'.format(num_classes))

    # load label names
    train_label_names = train_dataset.obtain_label_names()
    label_names = train_label_names
    label_names_h5 = h5py.File(os.path.join(args.out_path, 'label_names.h5'), 'w')
    label_names_h5['y_names'] = label_names
    logger.info('The label names are: {}'.format(str(label_names)))

    return train_loader, label_names, num_classes, train_data_size


def train_val_net(net):
    """train and validation of the network"""
    time_start = time.time()
    train_num_batch = train_data_size / args.train_batch_size
    # for save training and validating process data
    train_loss_lst, val_loss_lst = [], []
    save_model_epoch = None
    for epoch in range(args.epoch):
        train_start_time = time.time()
        epoch += 1
        total_train_loss, total_val_loss = 0, 0
        # training
        for i, data in enumerate(train_loader, 0):
            # data loading
            points, labels = data
            # points[0]: [B, N, 3], points[1]: [B, N, 3] to points [2B, N, 3]
            points = torch.cat([points[0], points[1]], dim=0)
            points = points.transpose(2, 1)  # points [2B, 3, N]
            labels = labels[:, 0]  # [B,1] rank2 to [B] rank1
            bs = labels.shape[0]  # size in this batch, which is <=args.train_batch_size
            points, labels = points.to(device), labels.to(device)
            # forward process
            optimizer.zero_grad()
            net = net.train()
            features = net(points)
            # feat1 : (B, feat_dim) ftea2: (B, feat_dim)
            feat1, feat2 = torch.split(features, [bs, bs], dim=0)
            # features: (B, num_views, feat_dim); num_view is 2 here
            features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
            if args.contrastive_method == 'SupCon':
                loss = criterion(features, labels)
            elif args.contrastive_method == 'SimCLR':
                loss = criterion(features)
            else:
                raise ValueError('contrastive method not supported: {}. '
                                 'Please select from SupCon or SimCLR'. format(args.contrastive_method))
            # backward process
            loss.backward()
            optimizer.step()
            if args.scheduler == 'wucd':
                scheduler.step(epoch-1 + i/train_num_batch)
            # for calculating training loss
            total_train_loss += loss.item()
        if args.scheduler == 'step':
            scheduler.step()
        # train accuracy loss
        avg_train_loss = total_train_loss / float(train_num_batch)
        train_loss_lst.append(avg_train_loss)
        train_end_time = time.time()
        train_time = round(train_end_time-train_start_time, 2)
        logger.info('{} epoch [{}/{}] time: {}s train loss: {} '.format(
            script_name, epoch, args.epoch, train_time, round(avg_train_loss, 4)))
        # save weights regularly
        if epoch % args.save_step == 0:
            torch.save(net.state_dict(), '{}/epoch_{}_model.pth'.format(args.out_path, epoch))
            print('Save {}/epoch_{}_model.pth'.format(args.out_path, epoch))
            save_model_epoch = epoch
    # save the last weight
    if save_model_epoch is None or save_model_epoch != epoch:
        torch.save(net.state_dict(), '{}/epoch_{}_model.pth'.format(args.out_path, epoch))
    torch.save(net.state_dict(), '{}/last_model.pth'.format(args.out_path))
    # total processing time
    time_end = time.time()
    total_time = round(time_end-time_start, 2)
    logger.info('Total processing time is {}s'.format(total_time))


if __name__ == '__main__':
    # Variable Space
    parser = argparse.ArgumentParser(description="Train contrastive encoder in stage 2",
                                     epilog="Tengfei Xue txue4133@uni.sydney.edu.au")
    # Paths
    parser.add_argument('--input_path', type=str, default='./TrainData/outliers_data/DEBUG_kp0.1/h5_np15/',
                        help='Input graph data and labels')
    parser.add_argument('--out_path_base', type=str, default='./ModelWeights',
                        help='Save trained models')

    # parameters
    parser.add_argument('--k_fold', type=int, default=5, help='fold of cross-validation')
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--opt', type=str, required=True, help='type of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for Adam')
    parser.add_argument('--momentum', type=float, default=0, help='momentum for SGD')
    parser.add_argument('--scheduler', type=str, default='step', help='type of learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=40, help='Period of learning rate decay')
    parser.add_argument('--decay_factor', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--T_0', type=int, default=10, help='Number of iterations for the first restart (for wucd)')
    parser.add_argument('--T_mult', type=int, default=2, help='A factor increases Ti after a restart (for wucd)')
    parser.add_argument('--train_batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--save_step', type=int, default=5, help='The interval of saving weights')
    parser.add_argument('--eval_fold_zero', default=False, action='store_true', help='eval on fold 0, train on fold 1 2 3 4')

    # contrastive learning parameters
    parser.add_argument('--head_name', type=str, required=True, default='mlp', help="mlp | linear")
    parser.add_argument('--encoder_feat_num', type=int, required=True, default='128',
                        help='The output feature dimension of head for calculating the contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.1, required=True, help='The hyperparameter for contrastive loss')
    parser.add_argument('--contrastive_method', type=str, default='SupCon', help='Supcon is supervised method, SimCLR is unsupervised method')

    args = parser.parse_args()

    args.manualSeed = 0  # fix seed
    print("Random Seed: ", args.manualSeed)
    fix_seed(args.manualSeed)

    script_name = '<train_stage2_encoder>'

    args.input_path = unify_path(args.input_path)