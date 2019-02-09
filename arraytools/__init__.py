# coding: utf-8
"""
========
__init__
========

Script : __init__.py

Author : Dan.Patterson@carleton.ca

Modified : 2018-12-16

**Purpose**

Provide tools to facilitate working with numpy and geometry and attributes
largely derived from ArcMap and ArcGIS Pro.

Notes
-----

It is assumed throughout that numpy has been imported as

>>> import numpy as np

**Available modules and subpackages**
    Produced using the art.__art_modules__['create'] for a module.

_basic::

  arr_info, arr_info, keep_ascii, is_float, keep_nums, del_punc, even_odd,
  n_largest, n_smallest, num_to_mask, num_to_nan, pad_even_odd, pad_nan,
  pad_zero, reshape_options, shape_to2D

a_io::

  arr_json, dict_arrays, dict_struct, excel_np, iterable_dict, load_npy,
  load_txt, save_npy, save_txt, struct_dict

create::

  convex, circle, ellipse, hex_flat, hex_pointy, rectangle,
  triangle, pnt_from_dist_bearing, xy_grid, transect_lines

frmts::

  arr_info, del_punc, even_odd, is_float, keep_ascii, keep_nums, n_largest,
  n_smallest, num_to_mask, num_to_nan, pad_even_odd, pad_nan, pad_zero,
  reshape_options, shape_to2D

geom::

  _center, _centroid, _convert, _densify_2D, _extent, _flat_, _max, _min,
  _new_view_, _reshape_, _unpack, _view_, angle_2pnts, angle_np, angle_seq,
  angles_poly, areas, azim_np, centers, centroids, circle, densify,
  dist_bearing, dx_dy_np, e_area, e_dist, e_leng, ellipse, hex_flat,
  hex_pointy, lengths, radial_sort, rectangle, repeat, rotate, seg_lengths,
  segment, simplify, stride, total_length, trans_rot

grid::

   combine_, expand_zone, fill_arr, reclass_ranges, reclass_vals, scale_up

image::

  _even_odd, _pad_even_odd, _pad_nan, _pad_zero, a_filter, equalize,
  normalize, plot_img, rgb_gray

ndset::

  _view_as_, _check_dtype_, nd_diff, nd_diffxor, nd_intersect, nd_isin,
  nd_union, nd_uniq

py_tools::

  _flatten, combine_dicts, comp_info, dir_py, flatten_shape, folders,
  get_dir, pack, sub_folders, unpack

stackstats::

  check_shapes, check_stack, mask_stack, stack_cumprod, stack_cumsum,
  stack_max, stack_mean, stack_median, stack_min, stack_percentile,
  stack_prod, stack_stats, stack_stats_tbl, stack_std, stack_sum, stack_var

tbl::

  find_in, tbl_count, tbl_replace, tbl_sum

tblstats::

  _calc_stats, _numeric_fields_, col_stats, freq, group_stats, skew_kurt, summ

tools::

  _func, _tools_help_, arr2xyz, arrays_struct, block, block_arr, change_arr,
  concat_arrs, find, group_pnts, group_vals, is_in, make_blocks, make_flds,
  nd2rec, nd2struct, nd_rec, nd_struct, pack_last_axis, pad_, radial_sort,
  rc_vals, reclass, rolling_stats, running_count, scale, sequences,
  sliding_window_view, sort_cols_by_row, sort_rows_by_col, view_sort,
  split_array, stride, uniq, xy_vals

utils::

  _utils_help_, dirr, doc_func, get_func, get_modu, run_deco, time_deco,
  wrapper

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect



from . import (_basic, _io, create, frmts, geom, geom_common, geom_properties,
               grid, image, ndset, py_tools, saws, stackstats, surface,
               tbl, tblstats, tools, utils)

from ._basic import *
from ._io import load_npy, save_npy, load_txt, save_txt
from .create import *
from .frmts import prn
from .geom import *
from .geom_common import *
from .geom_properties import *
from .grid import *
from .ndset import *
from .saws import *
from .stackstats import *
from .tbl import find_a_in_b, find_in, _split_sort_slice_, tbl_count, tbl_sum
from .tools import *
from .utils import dirr

from .analysis import near

__art_version__ = "Arraytools version 1.0"
__art_all__ = ['__art_version__']
__art_dict__ = {
        '_basic': _basic.__all__,
        '_io': _io.__all__,
        'create': create.__all__,
        'frmts': frmts.__all__,
        'geom': geom.__all__,
        'geom_common': geom_common.__all__,
        'geom_properties': geom_properties.__all__,
        'grid': grid.__all__,
        'image': image.__all__,
        'ndset': ndset.__all__,
        'py_tools': py_tools.__all__,
        'saws': saws.__all__,
        'stackstats': stackstats.__all__,
        'tbl': tbl.__all__,
        'tblstats': tblstats.__all__,
        'tools': tools.__all__,
        'utils': utils.__all__
        }
#__art_all__.extend(__art_modules__)
#__all__mods__ = __art_modules__
#
#
#def __info__():
#    """information on the package.
#    To use, enter
#
#    >>> art.__info__()
#    """
#    from textwrap import wrap
#    d = __art_modules__
#    for k in d.keys():
#        vals = d[k]
#        vals.sort()
#        v = ", ".join([v for v in vals])
#        vl = wrap(v, 75)
#        vl = ["  " + v for v in vl]
#        print("\n{}::\n".format(k))
#        for i in range(len(vl)):
#            print("{}".format(vl[i]))
#    #return vl
print("arraytools imported")
if __name__ == '__main__':
    print('arraytools.__init__ ...')