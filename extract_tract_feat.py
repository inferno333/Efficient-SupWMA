import utils.tract_feat as tract_feat

import whitematteranalysis as wma
import numpy as np

import argparse
import os
import h5py


def gen_features():
    print(script_name, 'Computing feauture:', args.feature)
    if args.feature == 'RAS':
        feat_RAS = tract_feat.feat_RAS(pd_tract, number_of_points=args.numPoints)

        # Reshape from 3D (num of fibers, num of points, num of features) to 4D (num of fibers, num of points, num of features, 1)
        # The 4D array considers the input has only one channel (depth = 1)
        feat_shape = np.append(feat_RAS.shape, 1)
        feat = np.reshape(feat_RAS, feat_shape)

    elif args.feature == 'RAS-3D':

        feat_RAS_3D = tract_feat.feat_RAS_3D(pd_tract, number_of_points=args.numPoints, repeat_time=args.numRepeats)

        feat = feat_RAS_3D

    elif args.feature == 'RASCurvTors':

        feat_curv_tors = tract_feat.feat_RAS_curv_tors(pd_tract, number_of_points=args.numPoints)

        feat_shape = np.append(feat_curv_tors.shape, 1)

        feat = np.reshape(feat_curv_tors, feat_shape)

    elif args.feature == 'CurvTors':

        feat_curv_tors = tract_feat.feat_curv_tors(pd_tract, number_of_points=args.numPoints)

        feat_shape = np.append(feat_curv_tors.shape, 1)

        feat = np.reshape(feat_curv_tors, feat_shape)

    else:
        raise ValueError