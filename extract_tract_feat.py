import utils.tract_feat as tract_feat

import whitematteranalysis as wma
import numpy as np

import argparse
import os
import h5py


def gen_features():
    print(script_name, 'Computing feauture:', args.feature)
    if args.f