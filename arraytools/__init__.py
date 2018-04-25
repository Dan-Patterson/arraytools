# coding: utf-8
"""
Arraytools
==========

Script : __init__.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-04-17

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
['arr_json', 'array2raster', 'load_npy', 'rasters2nparray', 'read_txt',
 'save_npy', 'save_txt']

apt.py :
    tools for arcpy tools
>>> art.apt.__all__
['_arr_common', '_id_geom_array', '_split_array', 'arc_np', 'arr_pnts',
 'arr_polygon_fc', 'arr_polyline_fc', 'array_fc', 'array_struct', 'change_fld',
 'fc_array', 'obj_polyline', 'obj_polyline', 'pnts_arr', 'polygons_arr',
 'polylines_arr', 'shapes_fc', 'shapes_fc', 'struct_polygon',
 'struct_polyline', 'tbl_arr', 'to_fc']

fc.py :
    tools for working with featureclasses
>>> art.fc.__all__
 ['_cursor_array', '_geo_array', '_get_shapes', '_ndarray', '_props',
 '_two_arrays', '_xy', '_xyID', '_xy_idx', 'arrays_cols', 'change_fld',
 'concat_arrs', 'join_arr_fc', 'obj_array', 'orig_dest_pnts']

frmts.py :
    Format options for viewing of numpy arrays in a variety of ways.
>>> art.frmts.__all__
['_demo_frmt', '_demo_ma', '_demo_rec', 'col_hdr', 'deline', 'form_', 'frmt_',
 'frmt_ma', 'frmt_rec', 'in_by', 'make_row_format', 'pd_', 'redent']

geom :
    Geometry related functions
>>> art.geom.__all__
['_center', '_centroid', '_convert', '_densify_2D', '_extent', '_flat_',
 '_max', '_min', '_new_view_', '_reshape_', '_unpack', '_view_', 'angle_2pnts',
 'angle_np', 'angle_seq', 'angles_poly', 'areas', 'azim_np', 'centers',
 'centroids', 'circle', 'densify', 'dist_bearing', 'dx_dy_np', 'e_area',
 'e_dist', 'e_leng', 'ellipse', 'hex_flat', 'hex_pointy', 'lengths',
 'radial_sort', 'rectangle', 'repeat', 'rotate', 'seg_lengths', 'simplify',
 'total_length']

image :
    image related functions
>>> art.image.__all__
['_even_odd', '_pad_even_odd', '_pad_nan', '_pad_zero', 'a_filter', 'equalize',
 'normalize', 'plot_img', 'rgb_gray']

py_tools :
    Python, numpy and other stack generic functions:
>>> art.py_tools.__all__
['_flatten', 'dir_py', 'dirr', 'dirr2', 'flatten_shape', 'folders', 'get_dir',
 'pack', 'sub_folders', 'unpack']

tools.py :
    Main tool set containing the following functions...
>>> art.tools.__all__
 ['_func', '_help', '_pad_', 'arr2xyz', 'block', 'block_arr', 'change_arr',
 'doc_func', 'find', 'get_func', 'get_modu', 'group_pnts', 'group_vals',
 'info', 'is_in', 'make_blocks', 'make_flds', 'n_largest', 'n_smallest',
 'nd2struct', 'num_to_mask', 'num_to_nan', 'rc_vals', 'nd_rec', 'reclass',
 'rolling_stats', 'scale', 'sort_cols_by_row', 'sort_rows_by_col',
 'split_array', 'stride', 'uniq', 'xy_vals']


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
from textwrap import dedent, indent, wrap
# ---- import *.py scripts and functions ----
from . import tools
from .tools import *
from . import _common
from ._common import _describe, fc_info, fld_info, tweet
from . import py_tools
from .py_tools import *
from . import a_io
from .a_io import *
from . import apt
from .apt import *
from . import fc
from .fc import *
from . import frmts
from .frmts import *
from . import geom
from .geom import *
from . import image
from .image import *
#
# ---- imports from subfolders
from . import analysis
from .analysis import *
from . import geomtools
from .geomtools import circular, mesh_pnts, mst, n_spaced, pip
from . import graphing
from .graphing import plot_pnts_
from . import rasters
from .rasters import  conversion, grid, rasterstats, surface
from . import stats
from .stats.cross_tab import crosstab


def __art_modules__():
    """print the array tools modules"""
    frmt = """\
-----------------------------------------------------------------------
#Information on .... arraytools ....
#More information can be obtained for the following modules...
#Use ... dir(art.module) ... where 'module' is in...
#- _common\n....{}
#- a_io\n....{}
#- apt\n....{}
#- analysis\n....{}
#- fc\n....{}
#- frmts\n....{}
#- geom\n....{}
#- image\n...{}
#- tools\n....{}
#"""
#    print(dedent(frmt).format(*__args))

#
__art_version__ = "Arraytools version 1.0"
__all__ = ['__art_version__', '__art_modules__']
__mods__ = {'_common': _common.__all__,
            'a_io': a_io.__all__,
            'analysis': analysis.__all__,
            'apt': apt.__all__,
            'fc': fc.__all__,
            'frmts': frmts.__all__,
            'geom': geom.__all__,
            'image': image.__all__,
            'py_tools': py_tools.__all__,
            'tools': tools.__all__
            }

__all__.extend(__mods__)
#
#__all__.sort()
#del _arg
#del __args
