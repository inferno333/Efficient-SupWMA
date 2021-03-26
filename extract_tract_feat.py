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
        # The 4D array considers the input has only one channel