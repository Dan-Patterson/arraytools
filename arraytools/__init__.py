# coding: utf-8
"""
arraytools
==========

Script : __init__.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-11-23

**Purpose:**

Provide tools to facilitate working with numpy and geometry and attributes
largely derived from ArcMap and ArcGIS Pro.

Documentation notes
-------------------

It is assumed throughout that numpy has been imported as

>>> import numpy as np

**Available modules and subpackages**

Produced using _base_functions.art_info()

_base_functions::

- arr_info, arr_info, keep_ascii, is_float, keep_nums, del_punc, even_odd,
- n_largest, n_smallest, num_to_mask, num_to_nan, pad_even_odd, pad_nan,
- pad_zero, reshape_options, shape_to2D

a_io::

- arr_json, dict_arrays, dict_struct, excel_np, iterable_dict, load_npy,
- load_txt, save_npy, save_txt, struct_dict

frmts::

- arr_info, del_punc, even_odd, is_float, keep_ascii, keep_nums, n_largest,
- n_smallest, num_to_mask, num_to_nan, pad_even_odd, pad_nan, pad_zero,
- reshape_options, shape_to2D

geom::

- _center, _centroid, _convert, _densify_2D, _extent, _flat_, _max, _min,
- _new_view_, _reshape_, _unpack, _view_, angle_2pnts, angle_np, angle_seq,
- angles_poly, areas, azim_np, centers, centroids, circle, densify,
- dist_bearing, dx_dy_np, e_area, e_dist, e_leng, ellipse, hex_flat,
- hex_pointy, lengths, radial_sort, rectangle, repeat, rotate, seg_lengths,
- segment, simplify, stride, total_length, trans_rot

grid::

- combine_, expand_zone, fill_arr, reclass_ranges, reclass_vals, scale_up

image::

- _even_odd, _pad_even_odd, _pad_nan, _pad_zero, a_filter, equalize,
- normalize, plot_img, rgb_gray

ndset::

- _view_as_, is_in, nd_diff, nd_diffxor, nd_intersect, nd_union, nd_uniq

py_tools::

- _flatten, combine_dicts, comp_info, dir_py, flatten_shape, folders,
- get_dir, pack, sub_folders, unpack

stackstats::

- check_shapes, check_stack, mask_stack, stack_cumprod, stack_cumsum,
- stack_max, stack_mean, stack_median, stack_min, stack_percentile,
- stack_prod, stack_stats, stack_stats_tbl, stack_std, stack_sum, stack_var

tbl::

- find_in, tbl_count, tbl_replace, tbl_sum

tblstats::

- _calc_stats, _numeric_fields_, col_stats, freq, group_stats, skew_kurt, summ

tools::

- _func, _tools_help_, arr2xyz, arrays_struct, block, block_arr, change_arr,
- concat_arrs, find, group_pnts, group_vals, is_in, make_blocks, make_flds,
- nd2rec, nd2struct, nd_rec, nd_struct, pack_last_axis, pad_, radial_sort,
- rc_vals, reclass, rolling_stats, running_count, scale, sequences,
- sliding_window_view, sort_cols_by_row, sort_rows_by_col, split_array,
- stride, uniq, xy_vals

utils::

- _utils_help_, dirr, doc_func, get_func, get_modu, run_deco, time_deco,
- wrapper

================

**arraytools.analysis**

Tools for calculating distance, proximity, angles
- compass, line_dir, not_closer, n_near, vincenty

**arraytools.geomtools**

>>> from arraytools.geomtools import  `(either name or *)`

Special computational geometry tools, including
- circular, mesh_pnts, mst, n_spaced, pip

arraytools.graphing

Graphing capabilities using MatPlotLib as the basic graphing program
- `plot_pnts_`

**arraytools.stats**
  Statistics and related
- crosstab

**arraytools.other**
  Placeholder


examples:
  Documentation for *.py script, will have the same name but end with *.txt.


"""

print("arraytools imported")
from . import (_base_functions, a_io, frmts, geom, grid, image, py_tools,
               stackstats, surface, tbl, tblstats, tools, utils)

from ._base_functions import (arr_info, even_odd, n_largest,
                              n_smallest, num_to_mask, num_to_nan,
                              pad_even_odd, pad_nan, pad_zero,
                              reshape_options, shape_to2D)

from .a_io import load_npy, save_npy, load_txt, save_txt
from .frmts import prn
from .geom import *
from .grid import *
from .ndset import *
from .stackstats import *
from .tbl import find_in, tbl_count, tbl_sum
from .tools import *
from .utils import dirr

__art_version__ = "Arraytools version 1.0"
__art_all__ = ['__art_version__']
__art_modules__ = {
        '_base_functions': _base_functions.__all__,
        'a_io': a_io.__all__,
        'frmts': frmts.__all__,
        'geom': geom.__all__,
        'grid': grid.__all__,
        'image': image.__all__,
        'ndset': ndset.__all__,
        'py_tools': py_tools.__all__,
        'stackstats': stackstats.__all__,
        'tbl': tbl.__all__,
        'tblstats': tblstats.__all__,
        'tools': tools.__all__,
        'utils': utils.__all__
             }
__art_all__.extend(__art_modules__)
__all__ = [__art_modules__]


def _info():
    """information on the package.
    To use... art._info()
    """
    from textwrap import wrap
    d = __art_modules__
    for k in d.keys():
        vals = d[k]
        vals.sort()
        v = ", ".join([v for v in vals])
        vl = wrap(v, 75)
        vl = ["- " + v for v in vl]
        print("\n{}::\n".format(k))
        for i in range(len(vl)):
            print("{}".format(vl[i]))
    #return vl
# =============================================================================
# import numpy as np
# #
# # ---- import *.py scripts and functions ----
# #
# from .frmts import prn as prn
# from .utils import dirr as dirr
# print(locals().keys())
# from ._base_functions import (arr_info, n_largest, n_smallest,
#                               num_to_nan, num_to_mask)
# from . import a_io
# from .a_io import load_npy, save_npy, load_txt, save_txt
# from . import frmts
# from .frmts import *
# from . import geom
# from .geom import *
# from . import image
# from . import grid
# #from .grid import *
# from . import gridstats
# #from .gridstats import *
# #from .image import *
# from . import py_tools
# #from .py_tools import *
# from . import stats
# from .stats import cross_tab  #, field_stats, frequency
# from . import surface
# ##from .surface import *
# from .tools import *
# from . import tbl
# #from .tbl import (find_text, tbl_count, tbl_sum)
# from . import tblstats
# from . import utils
# from .utils import (doc_func, get_func, get_modu, dirr)
# from . import tools
#
# # ---- imports from subfolders/packages
# from . import analysis
# #from .analysis import *
# from . import fc_tools
# from . import geomtools
# #from .geomtools import circular, mesh_pnts, mst, n_spaced, pip
# from . import graphing
# #from .graphing import plot_pnts_
# from . import rasters
# #from .rasters import (conversion, tifffile, ascii_to_raster, raster_functions,
# #                      rasters, rasterstats)
#
# #from . import stats
# #from .stats.cross_tab import crosstab
# #
# ##
# _art_version__ = "Arraytools version 1.0"
# __art_all__ = ['__art_version__']
# __art_module_dict__ = {'_base_functions': _base_functions.__all__,
#             'a_io': a_io.__all__,
#             'frmts': frmts.__all__,
#             'geom': geom.__all__,
#             'grid': grid.__all__,
#             'gridstats': gridstats.__all__,
#             'image': image.__all__,
#             'py_tools': py_tools.__all__,
#             'tbl': tbl.__all__,
#             'tblstats': tblstats.__all__,
#             'tools': tools.__all__,
#             'utils': utils.__all__
#             }
# __art_packages__ = ['analysis', 'fc_tools', 'form', 'geomtools', 'graphing',
#                     'rasters']
# __art_all__.extend(__art_module_dict__)
# #
# __art_all__.sort()
# __art_all__.extend(__art_packages__)
# #del _arg
# #del __args
#
# =============================================================================
