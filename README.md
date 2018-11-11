
arraytools
==========

Provides tools to facilitate working with numpy and the geometry and attributes of spatial data.  The focus is largely on rasters and featureclasses for use within ArcGIS Pro.

Documentation notes
-------------------
It is assumed throughout that numpy has been imported as

import numpy as np

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

**data_maker.py**



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


**geom.py:**  (12)  Geometry related function

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

**grid.py:**

    1. 3D array functions
         combine_, check_shapes, check_stack, mask_stack,
    2. statistical functions
       stack_percentile, stack_sum, stack_cumsum, stack_prod, stack_cumprod, stack_min, stack_mean,
       stack_median, stack_max, stack_std, stack_var, stack_stats
    3. other functions
       expand_zone, fill_arr, reclass_vals, reclass_ranges, scale_up

**graphing:**  Graphing capabilities using MatPlotLib as the basic graphing program
     plot_pnts_

**grid_stats.py:**

**image.py:**

**py_tools.py:**

**surface.py:**

**tbl.py:**

**tools.py**  (30)    Main tool set containing the following functions...

:__all_art__:

     'doc_func', 'get_func', 'get_modu', 'info', 'num_to_nan', 'num_to_mask', 'make_blocks',
     'make_flds', 'rec_arr', 'arr2xyz', 'change_arr', 'nd2struct', 'scale', 'split_array', 'stride',
     '_pad_', 'block', 'block_arr', 'find', '_func', 'group_pnts', 'group_vals', 'reclass',
     'rolling_stats', 'uniq', 'is_in', 'n_largest', 'n_smallest', 'rc_vals', 'xy_vals',
     'sort_rows_by_col', 'sort_cols_by_row', '_help'
    

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
