
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