TOOLS
=====


Some of the functions in the `tools.py` script are demonstrated below.
Not all are.

Let's get started with hopefully a complete list

```
_help()

:-------------------------------------------------------------------:
: ---- arrtools functions  (loaded as 'art') ----
: ---- from tools.py
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
(10) change_arr(a, order=[], prn=False)
     reorder and/or drop columns
(11) nd2struct(a)
     convert an ndarray to a structured array with fields
(12) scale(a, x=2, y=2, num_z=None)
     scale an array up in size by repeating values
(13) split_array(a, fld='ID')
     split an array using an index field
(14) _pad_
(15) stride(a, r_c=(3, 3))
     stride an array for moving window functions
(16) make_blocks
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
(26) n_smallest(a, num=1, by_row=True)
(27) rc_vals
(28) xy_vals
(29) sort_rows_by_col
(30) sort_cols_by_row
(31) radial_sort
 ---  _help  this function
:-------------------------------------------------------------------:

```
-----

```
_demo_tools()

:----------------------------------------------------------------------
:---- doc_func(func) ----
:Code for a function on line...2075...
:
2075  def pyramid(core=9, steps=10, incr=(1, 1), posi=True):
2076      """Create a pyramid see pyramid_demo.py"""
2077      a = np.array([core])
2078      a = np.atleast_2d(a)
2079      for i in range(1, steps):
2080          val = core - i
2081          if posi and (val <= 0):
2082              val = 0
2083          a = np.lib.pad(a, incr, "constant", constant_values=(val, val))
2084      return a

Comments preceeding function
None
function?... True ... or method? False
Module name... None
Full specs....
('args', ['core', 'steps', 'incr', 'posi'])
('varargs', None)
('varkw', None)
('defaults', (9, 10, (1, 1), True))
('kwonlyargs', [])
('kwonlydefaults', None)
('annotations', {})
----------------------------------------------------------------------
```

-----

The arrays that are used in the following examples are defined below

```
: ----- _demo --------------------------------------------------------------
:
:Input ndarray, 'a' ...
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

:Input ndarray, 'b' ...
array([(0, 1,  2,  3), (4, 5,  6,  7), (8, 9, 10, 11)],
      dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')])

:Input ndarray, 'c' ...
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])

:Input ndarray, 'd' ...
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35],
       [36, 37, 38, 39, 40, 41],
       [42, 43, 44, 45, 46, 47],
       [48, 49, 50, 51, 52, 53]])

```
-----

The sample functions are listed by number below.

```
:---- Functions by number  ---------------------------------------------
:(9)  arr2xyz(a, verbose=False)
[[ 0  0  0]
 [ 1  0  1]
 [ 2  0  2]
 [ 3  0  3]
 [ 0  1  4]
 [ 1  1  5]
 [ 2  1  6]
 [ 3  1  7]
 [ 0  2  8]
 [ 1  2  9]
 [ 2  2 10]
 [ 3  2 11]]

:(17)  block_arr(a, win=[2, 2], nodata=-1)
[[[ 0  1]
  [ 4  5]]

 [[ 2  3]
  [ 6  7]]

 [[ 8  9]
  [-1 -1]]

 [[10 11]
  [-1 -1]]]

:(10) change_arr(b, order=['B', 'C', 'A'], prn=False
:    Array 'b', reordered with 2 fields dropped...
array([[(1,  2, 0)],
       [(5,  6, 4)],
       [(9, 10, 8)]],
      dtype=[('B', '<i4'), ('C', '<i4'), ('A', '<i4')])


:(5) scale() ... scale an array up by an integer factor...
[[ 0  0  1  1  2  2  3  3]
 [ 0  0  1  1  2  2  3  3]
 [ 4  4  5  5  6  6  7  7]
 [ 4  4  5  5  6  6  7  7]
 [ 8  8  9  9 10 10 11 11]
 [ 8  8  9  9 10 10 11 11]]


:(4) array info ... info(a)

:---------------------
:Array information....
: OWNDATA: if 'False', data are a view
:flags....
:     C_CONTIGUOUS : True
:     F_CONTIGUOUS : False
:     OWNDATA : False
:     WRITEABLE : True
:     ALIGNED : True
:     UPDATEIFCOPY : False
:array
:  |__shape (9, 6)
:  |__ndim  2
:  |__size  54
:  |__bytes 216
:  |__type  <class 'numpy.ndarray'>
:  |__strides  (24, 4)
:dtype      int32
:  |__kind  i
:  |__char  l
:  |__num   7
:  |__type  <class 'numpy.int32'>
:  |__name  int32
:  |__shape ()
:  |__description
:  |  |__name, itemsize
:     |__['', '<i4']
:---------------------


:(7) make_flds() ... create default field names ...
[('A', '<i8'), ('B', '<i8'), ('C', '<i8')]


:(13) split_array() ... split an array according to an index field
[array([(0, 1, 2, 3)],
      dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')]), array([(4, 5, 6, 7)],
      dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')]), array([(8, 9, 10, 11)],
      dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')])]


:(15) stride(a, (3, 3)) ... stride an array ....
[[[ 0  1  2]
  [ 4  5  6]
  [ 8  9 10]]

 [[ 1  2  3]
  [ 5  6  7]
  [ 9 10 11]]]


:(16) make_blocks(rows=3, cols=3, r=2, c=2, dt='int')
[[0 0 1 1 2 2]
 [0 0 1 1 2 2]
 [3 3 4 4 5 5]
 [3 3 4 4 5 5]
 [6 6 7 7 8 8]
 [6 6 7 7 8 8]]


:(11) nd2struct() ... make a structured array from another array ...
array([(0, 1,  2,  3), (4, 5,  6,  7), (8, 9, 10, 11)],
      dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')])


:(22) rolling_stats()... stats for a strided array ...
:    min, max, mean, sum, std, var, ptp
(0, 53, 26.5, 26.5, 1431, 15.58578412100805, 242.91666666666666, 53)


```
