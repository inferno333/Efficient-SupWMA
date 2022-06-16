"""Reference from https://github.com/zhangfanmark/DeepWMA"""
import numpy as np
import whitematteranalysis as wma
import utils.fibers as fibers


def feat_RAS(pd_tract, number_of_points=15):
    """The most simple feature for initial test"""

    fiber_array = wma.fibers.FiberArray()
    fiber_array.convert_from_polydata(pd_tract, points_per_fiber=number_of_points)
    # fiber_array_r, fiber_array_a, fiber_array_s have the same size: [number of fibers, points of each fiber]
    feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s))

    return feat


def feat_curv_tors(pd_tract, number_of_points=15):
    """The most simple feature for initial test"""

    fiber_array = fibers.FiberArray()
    fiber_array.convert_from_polydata_with_trafic(pd_tract, points_per_fiber=number_of_points)

    feat = np.dstack((fiber_array.fiber_array_cur, fiber_array.fiber_array_tor))

    return feat


def feat_RAS_curv_tors(pd_tract, number_of_points=15):
    """The most simple feature for initial test"""

    fiber_array = fibers.FiberArray()
    fiber_array.convert_from_polydata_with_trafic(pd_tract, points_per_fiber=number_of_points)

    feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s,
                      fiber_array.fiber_array_cur, fiber_array.fiber_array_tor))

    return feat


def feat_RAS_3D(pd_tract, number_of_points=15, repeat_time=15):
    """The most simple feature for initial test"""

    feat = feat_RAS(pd_tract,
                    number_of_points=number_of_points