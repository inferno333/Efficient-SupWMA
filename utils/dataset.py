
from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import h5py
import sys
import os
sys.path.append('../')
import utils.tract_feat as tract_feat


class SupConDataset(data.Dataset):
    """Obtain data from ORG dataset and then generate bilateral pair for each fiber"""
    # TODO: Feel free to change the data loading module to fit your data.
    def __init__(self, root, logger, num_fold=1, k=5, split='train'):
        self.root = root
        self.split = split
        self.num_fold = num_fold
        self.k = k
        self.logger = logger
        features_combine = None
        labels_combine = None
        if self.split == 'train':
            train_fold = 0
            train_fold_lst = []
            for i in range(self.k):
                if i+1 != self.num_fold:
                    # load feature data
                    feat_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_featMatrix_{}.h5'.format(str(i+1))), 'r')
                    features = np.concatenate((feat_h5['sc_feat'], feat_h5['other_feat']), axis=0)
                    # load label data
                    label_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_label_{}.h5'.format(str(i+1))), 'r')
                    labels = np.concatenate((label_h5['sc_label'], label_h5['other_label']), axis=0)
                    if train_fold == 0:
                        features_combine = features
                        labels_combine = labels
                    else:
                        features_combine = np.concatenate((features_combine, features), axis=0)
                        labels_combine = np.concatenate((labels_combine, labels), axis=0)
                    train_fold_lst.append(i+1)
                    train_fold += 1
            self.features = features_combine
            self.labels = labels_combine
            logger.info('use {} fold as train data'.format(train_fold_lst))
        else:
            # load feature data
            feat_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_featMatrix_{}.h5'.format(self.num_fold)), 'r')
            self.features = np.concatenate((feat_h5['sc_feat'], feat_h5['other_feat']), axis=0)
            # load label data
            label_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_label_{}.h5'.format(self.num_fold)), 'r')
            self.labels = np.concatenate((label_h5['sc_label'], label_h5['other_label']), axis=0)
            logger.info('use {} fold as validation data'.format(self.num_fold))

        # label names list
        self.label_names = [*label_h5['label_names']]
        self.logger.info('The size of feature for {} is {}'.format(self.split, self.features.shape))

    def __getitem__(self, index):
        point_set = self.features[index]
        label = self.labels[index]
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            print('Feature is not in float32 format')

        if label.dtype == 'int64':
            label = torch.from_numpy(np.array([label]))
        else:
            label = torch.from_numpy(np.array([label]).astype(np.int64))
            print('Label is not in int64 format')

        # bilateral pair for pointset
        point_set_bilateral = point_set.detach().clone()
        point_set_bilateral[:, 0] = -point_set_bilateral[:, 0]
        new_point_set = [point_set, point_set_bilateral]

        return new_point_set, label

    def __len__(self):
        return len(self.labels)

    def obtain_label_names(self):
        return self.label_names


class ORGDataset(data.Dataset):
    def __init__(self, root, logger, num_fold=1, k=5, split='train'):
        # TODO: Feel free to change the data loading module to fit your data.
        # TODO: I saved my data into .h5 file, the size of "features" is [num_samples, num_points, 3], and the size of "labels" is [num_samples, ]
        self.root = root
        self.split = split
        self.num_fold = num_fold
        self.k = k
        self.logger = logger
        features_combine = None
        labels_combine = None
        if self.split == 'train':
            train_fold = 0
            train_fold_lst = []
            for i in range(self.k):
                if i+1 != self.num_fold:
                    # load feature data
                    # TODO: Feel free to change the path
                    feat_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_featMatrix_{}.h5'.format(str(i+1))), 'r')
                    features = np.concatenate((feat_h5['sc_feat'], feat_h5['other_feat']), axis=0)
                    # load label data
                    label_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_label_{}.h5'.format(str(i+1))), 'r')
                    labels = np.concatenate((label_h5['sc_label'], label_h5['other_label']), axis=0)
                    if train_fold == 0:
                        features_combine = features
                        labels_combine = labels
                    else:
                        features_combine = np.concatenate((features_combine, features), axis=0)
                        labels_combine = np.concatenate((labels_combine, labels), axis=0)
                    train_fold_lst.append(i+1)
                    train_fold += 1
            self.features = features_combine
            self.labels = labels_combine
            logger.info('use {} fold as train data'.format(train_fold_lst))
        else:
            # load feature data
            # TODO: Feel free to change the path
            feat_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_featMatrix_{}.h5'.format(self.num_fold)), 'r')
            self.features = np.concatenate((feat_h5['sc_feat'], feat_h5['other_feat']), axis=0)
            # load label data
            label_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_label_{}.h5'.format(self.num_fold)), 'r')
            self.labels = np.concatenate((label_h5['sc_label'], label_h5['other_label']), axis=0)
            logger.info('use {} fold as validation data'.format(self.num_fold))

        # label names list
        self.label_names = [*label_h5['label_names']]
        self.logger.info('The size of feature for {} is {}'.format(self.split, self.features.shape))
        # if split == 'val':
        #     print('The label names are: {}'.format(self.label_names))
