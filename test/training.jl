using PyCall

py"""
import numpy
xs = numpy.ones((2, 3))
"""

@info xs