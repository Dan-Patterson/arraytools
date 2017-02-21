# coding: utf-8
"""
arrtools
=======

Provides
  1. tool to facilitate working with numpy and geometry and attributes largely
     focused on ArcMap and ArcGIS Pro

Documentation notes
-------------------
It is assumed throughout that numpy has been imported as
   >>> import numpy as np

Available subpackages
---------------------
analysis:
    Tools for calculating distance, proximity, angles.
formating
    Format options to facilitate viewing of numpy arrays in a variety of ways.
geom
    Geometry related function
graph
    Graphing capabilities using MatPlotLib as the basic graphing program
stats
    Statistics and related
other
    Placeholder
examples
    Documentation for *.py script, will have the same name but end with *.txt.


"""
# from __future__ import division, absolute_import, print_function

# print("Initialization arraytools...\n...{}\nInitial keys...\
#       \n...{}".format(__path__[0], list(locals().keys())))

__all__ = ['tools']

from .tools import *
from .tools import _help
from . import analysis
from .analysis import *
from . import format
from .format import *
from . import geom
from .geom import e_area, e_dist, e_leng
# print("Final keys...\n...{}".format(list(locals().keys())))
