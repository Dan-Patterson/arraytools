# coding: utf-8
"""
Arraytools
==========

Script : __init__.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-10-16

**Purpose:**

Provide tools to facilitate working with numpy and geometry and attributes
largely derived from ArcMap and ArcGIS Pro.

Documentation notes
-------------------

It is assumed throughout that numpy has been imported as
>>> import numpy as np

Available modules and subpackages
---------------------------------
a_io.py :
    io tools for numpy arrays and operating system access
>>>  art.a_io.__all__
load_npy, save_npy, read_txt, save_txt, arr_json

frmts.py :
    Format options for viewing of numpy arrays in a variety of ways.
>>> art.frmts.__all__
col_hdr, deline, in_by, redent, _chunks, frmt_, frmt_ma, frmt_rec,
pd_, make_row_format, form_

geom :
    Geometry related functions
>>> art.geom.__all__
_flat_, _unpack, segment, stride, _new_view_, _view_, _reshape_, _min,
_max, _extent, _center, _centroid, centers, centroids, e_area, e_dist,
e_leng, areas, lengths, total_length, seg_lengths, radial_sort,
dx_dy_np, angle_np, azim_np, angle_2pnts, angle_seq, angles_poly,
dist_bearing, _densify_2D, _convert, densify, simplify, rotate,
trans_rot, repeat, circle, ellipse, rectangle, hex_flat, hex_pointy

grid :
    raster related statics
>>> art.grid.__all__
combine_, check_shapes, check_stack, mask_stack, combine_,
stack_percentile, stack_sum, stack_cumsum, stack_prod, stack_cumprod,
stack_min, stack_mean, stack_median, stack_max, stack_std, stack_var,
stack_stats, expand_zone, fill_arr, reclass_vals, reclass_ranges,
scale_up

gridstats :
    raster stack statistics
>>> art.gridstats.__all__
check_shapes, check_stack, mask_stack, stack_sum, stack_cumsum,
stack_prod, stack_cumprod, stack_min, stack_mean, stack_median,
stack_max, stack_std, stack_var, stack_percentile, stack_stats,
stack_stats_tbl

image :
    image related functions
>>> art.image.__all__
_even_odd, _pad_even_odd, _pad_nan, _pad_zero, a_filter, plot_img,
rgb_gray, normalize, equalize

py_tools :
    Python, numpy and other stack generic functions:
>>> art.py_tools.__all__
comp_info, get_dir, folders, sub_folders, dir_py, _flatten,
flatten_shape, pack, unpack

surface :
    surface tools for 2D arrays
>>> art.surface.__all__
pad_a, kernels, stride, filter_a, slope_a, aspect_a, hillshade_a

tools :
    Main tool set containing the following functions...
>>> art.tools.__all__
_tools_help_, n_largest, n_smallest, num_to_nan, num_to_mask, arr2xyz,
make_blocks, group_vals, reclass, scale, split_array, make_flds,
nd_rec, nd_struct, nd2struct, nd2rec, rc_vals, xy_vals, arrays_cols,
change_arr, concat_arrs, pad_, stride, block, sliding_window_view,
block_arr, rolling_stats, _func, find, group_pnts, uniq, is_in,
running_count, sequences, sort_cols_by_row, sort_rows_by_col,
radial_sort, pack_last_axis

utils :
    Utilities
>>> art.utils.__all__
non_ascii, non_punc, time_deco, run_deco, doc_func, get_func,
get_modu, info, dirr, wrapper

**Folder tools**
================

analysis :
  Tools for calculating distance, proximity, angles.
>>> 'compass', 'line_dir', 'not_closer', 'n_near', 'vincenty'

geomtools :  from arraytools.geomtools import *** either name or *
  Special computational geometry tools, including:

>>> circular, mesh_pnts, mst, n_spaced, pip

graphing :
  Graphing capabilities using MatPlotLib as the basic graphing program
>>> plot_pnts_

stats :
  Statistics and related
>>> crosstab

other :
  Placeholder


examples :
  Documentation for *.py script, will have the same name but end with *.txt.


"""
import numpy as np
#
# ---- import *.py scripts and functions ----
#
from ._base_functions import (arr_info, n_largest, n_smallest,
                              num_to_nan, num_to_mask)
from . import utils
from .utils import (doc_func, get_func, get_modu, dirr)
from . import tools
from .tools import *
from . import a_io
from .a_io import load_npy, save_npy, load_txt, save_txt
from .tbl import (find_text, tab_count, tab_sum)
from . import frmts
from .frmts import *
from . import geom
from .geom import *
from . import image
from . import grid
#from .grid import *
from . import gridstats
#from .gridstats import *
#from .image import *
from . import py_tools
#from .py_tools import *
from . import stats
from .stats import cross_tab, field_stats, frequency
from . import surface
##from .surface import *
##
## ---- imports from subfolders
from . import analysis
from .analysis import *
from . import fc_tools
from . import geomtools
from .geomtools import circular, mesh_pnts, mst, n_spaced, pip
from . import graphing
#from .graphing import plot_pnts_
from . import rasters
from .rasters import conversion, tifffile
#from . import stats
#from .stats.cross_tab import crosstab
#
##
_art_version__ = "Arraytools version 1.0"
__all__ = ['__art_version__']
__mods__ = {'a_io': a_io.__all__,
            'analysis': analysis.__all__,
            'frmts': frmts.__all__,
            'geom': geom.__all__,
            'image': image.__all__,
            'py_tools': py_tools.__all__,
            'tools': tools.__all__,
            'utils': utils.__all__
            }
#
__all__.extend(__mods__)
#
__all__.sort()
#del _arg
#del __args
