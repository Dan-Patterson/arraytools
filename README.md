
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
=======

Provides tools to facilitate working with numpy and geometry and attributes
largely derived from ArcMap and ArcGIS Pro.

Documentation notes
-------------------
It is assumed throughout that numpy has been imported as

import numpy as np

Available modules and subpackages
---------------------------------

**art.dirr(art)**

  
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
    
**_help()**
-------------------------------------------------------------------:
---- arrtools functions  (loaded as 'art') ----
---- from tools.py

(1)  doc_func(func=None)
     documenting code using inspect
     
(2)  get_func(obj, line_nums=True, verbose=True)
     pull in function code
     
(3)  get_modu(obj)
     pull in module code

(4)  info(a)  array info

(5a, b) num_to_nan, num_to_mask

(6)  make_blocks(rows=3, cols=3, r=2, c=2, dt='int')
     make arrays consisting of blocks

(7)  make_flds(n=1, as_type='float', names=None, def_name='col')
     make structured/recarray fields

(8)  rec_arr(a, flds=None, types=None)

(9)  arr2xyz(a, verbose=False)
     array (col, rows) to (x, y) and array values for z.

(10) nd2struct(a)
     convert an ndarray to a structured array with fields

(11) change(a, order=[], prn=False)
     reorder and/or drop columns

(12) scale(a, x=2, y=2, num_z=None)
     scale an array up in size by repeating values

(13) split_array(a, fld='ID')
     split an array using an index field

(14) _pad_

(15) stride(a, r_c=(3, 3))
     stride an array for moving window functions

(16) block

(17) block_arr(a, win=[3, 3], nodata=-1)
     break an array up into blocks

(18)  find(a, func, this=None, count=0, keep=[], prn=False, r_lim=2)
     find elements in an array using...
     func - (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
           (      , ==,  !=,  <,   <=,  >,   >=,  >a<, =>a<=,  <a> )

(19)  group_pnts(a, key_fld='ID', keep_flds=['X', 'Y', 'Z'])

(20)  group_vals(seq, delta=1, oper='!=')

(21) reclass(a, bins=[], new_bins=[], mask=False, mask_val=None)
     reclass an array

(22) rolling_stats((a0, no_null=True, prn=True))

(23) uniq(ar, return_index=False, return_inverse=False,
          return_counts=False, axis=0)

(24) is_in

(25) n_largest(a, num=1, by_row=True)

(26)    n_smallest(a, num=1, by_row=True)

(27) rc_vals

(28) xy_vals

(29) sort_rows_by_col

(30)sort_cols_by_row
: ---  _help  this function
:-------------------------------------------------------------------:
 
 

**analysis:**  (5)
    Tools for calculating distance, proximity, angles.
:__all__
  'compass', 'line_dir', 'not_closer', 'n_near', 'vincenty'

**geom:**  (12)
  Geometry related function
:__all_geo__
    '_view_', '_reshape_', 'areas', 'center', 'centroid',  'e_area',
    'obj_array', 'e_dist', 'e_leng', 'seg_lengths', 'total_length', 'lengths'

**graphing:**

  Graphing capabilities using MatPlotLib as the basic graphing program
     plot_pnts_

**stats:**

  Statistics and related
    crosstab

**other:**
    Placeholder

examples:
    Documentation for *.py script, will have the same name but end with *.txt.
