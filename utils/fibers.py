
""" fibers.py
This module contains code for representation of tractography using a
fixed-length parameterization.
class FiberArray
"""

import numpy
import vtk
import time


class Fiber:
    """A class for fiber tractography data, represented with a fixed length"""

    def __init__(self):
        self.r = None
        self.a = None
        self.s = None
        self.points_per_fiber = None
        self.hemisphere_percent_threshold = 0.95

    def get_equivalent_fiber(self):
        """ Get the reverse order of current line (trajectory), as the
        fiber can be equivalently represented in either order."""

        fiber = Fiber()

        fiber.r = self.r[::-1]
        fiber.a = self.a[::-1]
        fiber.s = self.s[::-1]

        fiber.points_per_fiber = self.points_per_fiber

        return fiber

    def get_reflected_fiber(self):
        """ Returns reflected version of current fiber by reflecting
        fiber across midsagittal plane. Just sets DeepWMAOutput R coordinate to -R."""

        fiber = Fiber()

        fiber.r = - self.r
        fiber.a = self.a
        fiber.s = self.s

        fiber.points_per_fiber = self.points_per_fiber

        return fiber

    def match_order(self, other):
        """ Reverse order of fiber to match this one if needed """
        # compute correlation
        corr = numpy.multiply(self.r, other.r) + \
               numpy.multiply(self.a, other.a) + \
               numpy.multiply(self.s, other.s)

        other2 = other.get_equivalent_fiber()
        corr2 = numpy.multiply(self.r, other2.r) + \
                numpy.multiply(self.a, other2.a) + \
                numpy.multiply(self.s, other2.s)

        if numpy.sum(corr) > numpy.sum(corr2):
            return other
        else:
            return other2

    def __add__(self, other):
        """This is the + operator for fibers"""
        other_matched = self.match_order(other)
        fiber = Fiber()
        fiber.r = self.r + other_matched.r
        fiber.a = self.a + other_matched.a
        fiber.s = self.s + other_matched.s
        return fiber

    def __div__(self, other):
        """ This is to divide a fiber by a number"""
        fiber = Fiber()
        fiber.r = numpy.divide(self.r, other)
        fiber.a = numpy.divide(self.a, other)
        fiber.s = numpy.divide(self.s, other)
        return fiber

    def __mul__(self, other):
        """ This is to multiply a fiber by a number"""
        fiber = Fiber()
        fiber.r = numpy.multiply(self.r, other)
        fiber.a = numpy.multiply(self.a, other)
        fiber.s = numpy.multiply(self.s, other)
        return fiber

    def __subtract__(self, other):
        """This is the - operator for fibers"""
        other_matched = self.match_order(other)
        fiber = Fiber()
        fiber.r = self.r - other_matched.r
        fiber.a = self.a - other_matched.a
        fiber.s = self.s - other_matched.s
        # fiber.r = self.r + other_matched.r
        # fiber.a = self.a + other_matched.a
        # fiber.s = self.s + other_matched.s
        return fiber


class FiberArray:
    """A class for arrays of fiber tractography data, represented with
    a fixed length"""

    def __init__(self):
        # parameters
        self.points_per_fiber = 10
        self.verbose = 0

        # fiber data
        self.fiber_array_r = None
        self.fiber_array_a = None
        self.fiber_array_s = None

        self.fiber_array_fs = None

        # DeepWMAOutput arrays indicating hemisphere/callosal (L,C,R= -1, 0, 1)
        self.fiber_hemisphere = None
        self.hemispheres = False
        self.hemisphere_percent_threshold = 0.95

        # DeepWMAOutput boolean arrays for each hemisphere and callosal fibers
        self.is_left_hem = None
        self.is_right_hem = None
        self.is_commissure = None

        # DeepWMAOutput indices of each type above
        self.index_left_hem = None
        self.index_right_hem = None
        self.index_commissure = None
        self.index_hem = None

        # DeepWMAOutput totals of each type also
        self.number_of_fibers = 0
        self.number_left_hem = None
        self.number_right_hem = None
        self.number_commissure = None

    def __str__(self):
        DeepWMAOutput = "\n points_per_fiber\t" + str(self.points_per_fiber) \
                        + "\n number_of_fibers\t\t" + str(self.number_of_fibers) \
                        + "\n fiber_hemisphere\t\t" + str(self.fiber_hemisphere) \
                        + "\n verbose\t" + str(self.verbose)

        return DeepWMAOutput

    def _calculate_line_indices(self, input_line_length, DeepWMAOutput_line_length):
        """ Figure out indices for downsampling of polyline data.
        The indices include the first and last points on the line,
        plus evenly spaced points along the line.  This code figures
        out which indices we actually want from a line based on its
        length (in number of points) and the desired length.
        """

        # this is the increment between DeepWMAOutput points
        step = (input_line_length - 1.0) / (DeepWMAOutput_line_length - 1.0)

        # these are the DeepWMAOutput point indices (0-based)
        ptlist = []
        for ptidx in range(0, DeepWMAOutput_line_length):
            # print(ptidx*step)
            ptlist.append(ptidx * step)

        # test
        if __debug__:
            # this tests we DeepWMAOutput the last point on the line
            # test = ((DeepWMAOutput_line_length - 1) * step == input_line_length - 1)
            test = (round(ptidx * step) == input_line_length - 1)
            if not test:
                print
                "<fibers.py> ERROR: fiber numbers don't add up."
                print
                step
                print
                input_line_length
                print
                DeepWMAOutput_line_length
                print
                test
                raise AssertionError

        return ptlist

    def get_fiber(self, fiber_index):
        """ Return fiber number fiber_index. Return value is class
        Fiber."""

        fiber = Fiber()
        fiber.r = self.fiber_array_r[fiber_index, :]
        fiber.a = self.fiber_array_a[fiber_index, :]
        fiber.s = self.fiber_array_s[fiber_index, :]

        fiber.points_per_fiber = self.points_per_fiber

        return fiber

    def get_equivalent_fiber(self, fiber_index):