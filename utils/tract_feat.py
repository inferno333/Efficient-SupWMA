"""Reference from https://github.com/zhangfanmark/DeepWMA"""
import numpy as np
import whitematteranalysis as wma
import utils.fibers as fibers


def feat_RAS(pd_tract, number_of_points=15):
    """The most simple feature for initial test"""

    fiber_array = wma.fibers.FiberArray()
    fiber_array.convert_from_polydata(pd_trac