
arraytools
==========

Provides tools to facilitate working with numpy and the geometry and attributes of spatial data.  The focus is largely on array/raster/grid data and vector geometry (ie featureclasses in ArcGIS Pro).

Structure
----------
A useful ditty to get a string representation of functions in a module.  
```
import arraytools as art
print(", ".join([i.replace("'", "") for i in dir(art.tools)])  # art.(script/module to import)
```

----
- **arraytools**  **...Tools for working with numpy arrays**
    - [`_base_functions.py`](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/_base_functions.py)
      - arr_info, even_odd, n_largest, n_smallest, np, num_to_mask, num_to_nan, pad_even_odd, pad_nan, pad_zero, reshape_options, shape_to2D, type_keys, type_vals
    - [aio.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/aio.py)
      - arr_info, even_odd, n_largest, n_smallest, num_to_mask, num_to_nan, pad_even_odd, pad_nan, pad_zero, reshape_options, shape_to2D, excel_np
    - [frmts.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/frmts.py)
      - \_check, \_chunks, \_col_format, \_col_kind_width, \_data, \_row_format, \_slice_cols, \_slice_head_tail, \_slice_rows, col_hdr, deline,  head_tail, in_by, make_row_format, pd_, prn, prn_, prn_3d4d, prn_ma, prn_nd, prn_q, prn_rec, prn_struct, quick_prn
    - [geom.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/geom.py)
      -  \_arrs_, \_center, \_centroid, \_convert, \_densify_2D, \_extent, \_flat_, \_max, \_min, \_new_view_, \_reshape_, \_unpack, \_view_, adjacency_edge, angle_2pnts, angle_between, angle_np, angle_seq, angles_poly, areas, as_strided, azim_np, centers, centroids, circle, densify, dist_bearing, dx_dy_np, e_area, e_dist, e_leng, ellipse, hex_flat, hex_pointy, intersect_pnt, lengths, pnt_, radial_sort, rectangle, repeat, rotate, seg_lengths, segment, simplify, stride, total_length, trans_rot
    - [grid.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/grid.py)
      - check_shapes, check_stack, combine_, expand_zone, fill_arr, mask_stack, nd2struct, reclass_ranges, reclass_vals, scale_up, stack_cumprod, stack_cumsum, stack_max, stack_mean, stack_median, stack_min, stack_percentile, stack_prod, stack_stats, stack_std, stack_sum, stack_var, stride
    - [gridstats.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/gridstats.py)
      - check_shapes, check_stack, mask_stack, stack_cumprod, stack_cumsum, stack_max, stack_mean, stack_median, stack_min, stack_percentile, stack_prod, stack_stats, stack_stats_tbl, stack_std, stack_sum, stack_var
    - [image.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/image.py)
      -  \_even_odd, \_pad_even_odd, \_pad_nan, \_pad_zero, a_filter, block, equalize, normalize, plot_img, rgb_gray, stride
    - [py_tools.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/py_tools.py)
      -  \_flatten, combine_dicts, comp_info, dir_py, flatten_shape, folders, get_dir, pack, sub_folders, sys, unpack
    - [surface.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/surface.py)
      - a2z, all_f, angle2azim, aspect_a, aspect_dem, circ_demo, circle_a, cross_f, dedent, filter_a, hillshade_a, kernels, no_cnt, pad_a, plot_, plt, plus_f, pyramid, slope_a, stride, surface_kernel
    - [tbl.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/tbl.py)
      - find_text, prn_struct, replace_, struct_deco, tab_count, tab_sum
    - [tools.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/tools.py)
      -  _func, \_tools_help_, arr2xyz, arrays_struct, as_strided, block, block_arr, change_arr, concat_arrs, find, group_pnts, group_vals, is_in, make_blocks, make_flds, nd2rec, nd2struct, nd_rec, nd_struct, pack_last_axis, pad_, pyramid, radial_sort, rc_vals, reclass, rolling_stats, running_count, scale, sequences, sliding_window_view, sort_cols_by_row, sort_rows_by_col, split_array, stride, uniq,  xy_vals
    - [utils.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/utils.py)
      - \_utils_help_, is_ascii, is_float, keep_nums, del_punc, del_punc_space, dirr, doc_func, get_func, get_modu, run_deco, time_deco, warnings, wrapper
  
----
- **. . . \analysis**
    - [array_moving.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/analysis/arr_moving.py)
      - `more to come`
----
- **. . . \fc_tools**  Many functions in this section require ArcGIS Pro be installed to access `arcpy`
    - [`_common.py`](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/fc_tools/_common.py)
      -  \_describe, arr_csv, de_punc, fc_info, fld_info, null_dict, tbl_arr, tweet'
    - [apt.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/fc_tools/apt.py)
      -  \_arr_common, \_id_geom_array, \_split_array, arc_np, arr_pnts, arr_polygon_fc, arr_polyline_fc, array_fc, array_struct, change_fld, dedent, fc_array, fc_info, obj_polygon, obj_polyline, output_points, output_polygons, output_polylines, pnts_arr, polygons_arr, polylines_arr, shapes_fc, struct_polygon, struct_polyline,  tbl_arr, to_fc, tweet'
    - [arc_io.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/fc_tools/arc_io.py)
      - array2raster, rasters2nparray,
      - `more to come`
    - [fc.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/fc_tools/fc.py)
      -  \_cross_3pnts, \_cursor_array, \_geo_array, \_get_shapes, \_ndarray, \_props, \_two_arrays, \_xy, \_xyID, \_xy_idx, change_fld, fc_info, ft, indent, join_arr_fc,obj_array, orig_dest_pnts, tweet, warnings'

----
- **. . . \geomtools**
    - [circular.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/geomtools/circular.py)
      - plot_, rot_matrix, \_arc, \_circle, arc_sector buffer_ring
    - [hulls.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/geomtools/hulls.py)
      - concave, convex
    - [mesh_pnts.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/geomtools/mesh_pnts.py)
    - [mst.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/geomtools/mst.py)
    - [n_spaced.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/geomtools/n_spaced.py)
    - [pip.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/geomtools/pip.py)
    - [split_polys.py](https://github.com/Dan-Patterson/arraytools/blob/master/arraytools/geomtools/split_polys.py)
      - `more to come`
 
----
- **. . . \graphing**
  
----
- **. . . \rasters**
 
----
- **. . . \stats**


Documentation notes
-------------------
It is assumed throughout that numpy has been imported as

`import numpy as np`

Available modules and subpackages
---------------------------------
 
**a_io.py**  (9)   io tools for numpy arrays

    1.  load_npy      - load numpy npy files
    2.  save_npy      - save array to *.npy format
    3.  read_txt      - read array created by save_txtt
    4.  save_txt      - save array to npy format
    5.  arr_json      - save to json format
    6.  dict_arrays   - dictionary to arrays
    7.  iterable_dict - iterable to dictionary
    8.  dict_struct   - dictionary to structured array
    9.  struct_dict   - structured array to dictionary
    10. excel_np      - excel xls/xlsx to numpy structured array using xlrd

**data_maker.py**
    Various functions for creating data.  Functions for numeric and text data conforming to 
    random sampling distributions. 

**frmts.py**  (18)    Format options to facilitate viewing of numpy arrays in a variety of ways.

    1.  col_hdr           column headers
    2.  deline            remove excessive blank lines
    3.  in_by             an indent variant with options 
    4.  redent            indent
    5.  _chunks           take chunks of stuff
    6.  head_tail         return the head/tail of a 1d array
    7.  _check            helper functions
    8.  _slice_rows
    9.  _slice_cols
    10. _slice_head_tail
    11. _col_format        printing section
    12. prn_nd             for ndarray
    13. prn_ma             for masked arrays
    14. prn_rec pd_        record/structured arrays
    15. prn_struct
    16. make_row_format    a big helper function
    17. prn_               ndarray variant
    18. prn         ---- this def is used to call all the others ----


**geom.py**  (40)  Geometry related functions

    1. helpers
       _flat_, _unpack, segment, stride, _new_view_, _view_, _reshape_,
    2. boundary
       _min, _max, _extent,
    3. centrality
       _center, _centroid, centers, centroids,
    4. area, length, distance
       e_area, e_dist, e_leng, areas, lengths, total_length, seg_lengths,
    5. sorting
       radial_sort
    6. angles
       dx_dy_np, angle_np, azim_np, angle_2pnts, angle_seq, angles_poly, dist_bearing,
    7. densify, simplify
       _densify_2D, _convert, densify, simplify,
    8. property alteration
       rotate,  trans_rot, repeat,
    9. construction
       circle, ellipse, rectangle, hex_flat, hex_pointy

**grid.py:**  ()
    Reclassification, array restructuring

    1. 3D array helper functions
         combine_, check_shapes, check_stack, mask_stack,
    2. moving to grid_stats below!!!
    3. other functions
       expand_zone, fill_arr, reclass_vals, reclass_ranges, scale_up

**graphing:**  Graphing capabilities using MatPlotLib as the basic graphing program
     plot_pnts_

**grid_stats.py:**
    Statistical functions for 3D-stacks of array data, like annual temperature data etc.
    Whatever can be stacked and is numeric.... you can get the stats for.  Moving window
    functions are also contained for all stats.

    1. 3D array helper functions
       check_shapes, check_stack, mask_stack
    2. statistical functions
       stack_sum, stack_cumsum, stack_prod, stack_cumprod, stack_min, stack_mean,
       stack_median, stack_max, stack_std, stack_var, stack_percentile, stack_stats,
       stack_stats_tbl

**image.py:**

    1. helper functions
       _even_odd, _pad_even_odd, _pad_nan, _pad_zero,
    2. image processing filtering
       a_filter
    3. graphing
       plot_img',
    4. conversion/alteration
       rgb_gray, normalize, equalize
       

**py_tools.py:**

    1. computer information
       comp_info
    2. basic folder functions
       get_dir, folders, sub_folders
    3. object and directory functions
       dir_py, _flatten, flatten_shape
    4. iterables
       pack, unpack, combine_dicts


**surface.py:**

**tbl.py:**

**tools.py**  (30)    Main tool set containing the following functions...

:__all_art__:

     'doc_func', 'get_func', 'get_modu', 'info', 'num_to_nan', 'num_to_mask', 'make_blocks',
     'make_flds', 'rec_arr', 'arr2xyz', 'change_arr', 'nd2struct', 'scale', 'split_array', 'stride',
     '_pad_', 'block', 'block_arr', 'find', '_func', 'group_pnts', 'group_vals', 'reclass',
     'rolling_stats', 'uniq', 'is_in', 'n_largest', 'n_smallest', 'rc_vals', 'xy_vals',
     'sort_rows_by_col', 'sort_cols_by_row', '_help'
    
## under construction below##
----------------------------------------------------------------------------

**analysis:**  (5)  Tools for calculating distance, proximity, angles.
    
:__all__
  'compass', 'line_dir', 'not_closer', 'n_near', 'vincenty'

**stats:**   Statistics and related
    crosstab

**other:**
    Placeholder

**fc.py**  (11)     tools for working with featureclasses
 :__all_fc__

    '_get_shapes', '_ndarray', '_props', '_two_arrays', '_xy',
    '_xyID', '_xy_idx', 'change_fld'
examples:
    Documentation for *.py script, will have the same name but end with *.txt.
