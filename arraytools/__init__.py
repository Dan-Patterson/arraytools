# coding: utf-8
"""
Modified: 2017-09-29

arrtools
=======

Provides tools to facilitate working with numpy and geometry and attributes
largely derived from ArcMap and ArcGIS Pro.

Documentation notes
-------------------
It is assumed throughout that numpy has been imported as
   >>> import numpy as np

Available modules and subpackages
---------------------
a_io.py
    io tools for numpy arrays and operating system access

frmts.py
    Format options to facilitate viewing of numpy arrays in a variety of ways.
 - 'col_hdr', 'deline', 'frmt_', 'frmt_ma', 'frmt_rec', 'frmt_struct',
   'in_by', 'make_row_format', 'redent'

tools.py
    Main tool set containing the following functions...
 - 'block_arr', 'change', 'doc_func', 'find', 'get_func', 'get_modu',
   'info', 'make_blocks', 'make_flds', 'nd_struct', 'reclass', 'scale',
   'stride', 'rolling_stats'
 - '_func', '_check', '_demo', 'run_deco', 'time_deco'

analysis:
    Tools for calculating distance, proximity, angles.

geom:
    Geometry related function
graphing:
    Graphing capabilities using MatPlotLib as the basic graphing program
stats:
    Statistics and related
other:
    Placeholder
examples:
    Documentation for *.py script, will have the same name but end with *.txt.


"""

from .a_io import  *
from .frmts import *
from .tools import *
from .tools import _help

from . import analysis
from . import geom
from . import graphing
from . import stats
from .analysis import *
from .geom import e_area, e_dist, e_leng, mst
from .graphing import plot_pnts_
from .stats.cross_tab import crosstab


__all__ = frmts.__all__
__all__ += tools.__all__
__all__ += analysis.__all__
#__all__ += geom.__all__

