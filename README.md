
***Arraytools***
================

2018-09-16 - Beginning of the reorganization of arraytools to separate the purely array functionality from functionality that uses arcpy


**Purpose**
I have consolidate various array scripts into one location.

Eventually it will become a package when I get done with it.

Some information from __init__.py
---------------------------------

Modified: 2017-11-04

arraytools
==========

Provides tools to facilitate working with numpy and geometry and attributes
largely derived from ArcMap and ArcGIS Pro.

Documentation notes
-------------------
It is assumed throughout that numpy has been imported as

import numpy as np

Available modules and subpackages
---------------------------------

 
**a_io.py**  (5)   io tools for numpy arrays

:__all_aio__

    1.  load_npy    - load numpy npy files
    2.  save_npy    - save array to *.npy format
    3.  read_txt    - read array created by save_txtt
    4.  save_txt    - save array to npy format
    5.  arr_json    - save to json format

**apt.py  (15)**    tools for arcpy tools

:__all_apt__

    '_arr_common', '_shapes_fc', 'arr_pnts', 'arr_polygon', 'arr_polyline',
    'array_fc', 'array_struct', 'change_fld', 'fc_array', 'pnts_arr',
    'polygons_arr', 'polylines_arr', 'tbl_arr', 'to_fc', 'tweet'

**fc.py**  (11)     tools for working with featureclasses
 
:__all_fc__

    '_get_shapes', '_ndarray', '_props', '_two_arrays', '_xy',
    '_xyID', '_xy_idx', 'change_fld'

**frmts.py**  (11)    Format options to facilitate viewing of numpy arrays in a variety of ways.

:__all_frmt__

    'col_hdr', 'deline', 'frmt_', 'frmt_ma', 'frmt_rec', 'frmt_struct',
    'in_by', 'make_row_format', 'redent', '_demo', '_ma_demo']

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

**geom:**  (12)  Geometry related function
  
:__all_geo__
    '_view_', '_reshape_', 'areas', 'center', 'centroid',  'e_area',
    'obj_array', 'e_dist', 'e_leng', 'seg_lengths', 'total_length', 'lengths'

**graphing:**  Graphing capabilities using MatPlotLib as the basic graphing program
     plot_pnts_

**stats:**   Statistics and related
    crosstab

**other:**
    Placeholder

examples:
    Documentation for *.py script, will have the same name but end with *.txt.
