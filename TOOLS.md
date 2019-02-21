TOOLS
=====


Some of the functions in the `tools.py` script are demonstrated below.
Not all are.

Let's get started with hopefully a complete list

```
__all__ = ['arr2xyz', 'make_blocks',     # (1-6) ndarrays ... make arrays,
           'group_vals', 'reclass',      #     change shape, arangement
           'scale', 'split_array',
           'make_flds', 'nd_rec',        # (7-14) structured/recdarray
           'nd_struct', 'nd2struct',
           'nd2rec', 'rc_vals', 'xy_vals',
           'arrays_struct',
           'change_arr', 'concat_arrs',  # (15-16) change/modify arrays
           'pad_', 'stride', 'block',    # (17-22) stride, block and pad
           'sliding_window_view',
           'block_arr', 'rolling_stats',
           '_func', 'find', 'find_closest',  # (23-28) querying, analysis
           'group_pnts',
           'uniq', 'is_in',
           'running_count', 'sequences',
           'pack_last_axis'  # extras -------
           ]
```

**arr2xyz**

Converts an array to xyz values.  The x, y values are derived from the row-column locations.
```
a = np.arange(12).reshape(3,4)

arr2xyz(a, keep_masked=False, verbose=False)  # verbose option False
 
array([[ 0,  0,  0],
       [ 1,  0,  1],
       [ 2,  0,  2],
       [ 3,  0,  3],
       [ 0,  1,  4],
       [ 1,  1,  5],
       [ 2,  1,  6],
       [ 3,  1,  7],
       [ 0,  2,  8],
       [ 1,  2,  9],
       [ 2,  2, 10],
       [ 3,  2, 11]])

arr2xyz(a, keep_masked=False, verbose=True)  # verbose option True

----------------------------
Meshgrid demo: array to x,y,z table
:Formulation...
:  XX,YY = np.meshgrid(np.arange(x.shape[1]),np.arange(x.shape[0]))
:Input table
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
:Raveled array, using x.ravel()
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
:XX in mesh: columns shape[1]
array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
:YY in mesh: rows shape[0]
array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
:Output:
array([[ 0,  0,  0],
       [ 1,  0,  1],
       [ 2,  0,  2],
       [ 3,  0,  3],
       [ 0,  1,  4],
       [ 1,  1,  5],
       [ 2,  1,  6],
       [ 3,  1,  7],
       [ 0,  2,  8],
       [ 1,  2,  9],
       [ 2,  2, 10],
       [ 3,  2, 11]])
:-----------------------------

```
**make_blocks**

Construct an array of patterns, with rows-columns, repeats and dtype,.

```
make_blocks(rows=3, cols=3, r=2, c=2, dt='int')

array([[0, 0, 1, 1, 2, 2],
       [0, 0, 1, 1, 2, 2],
       [3, 3, 4, 4, 5, 5],
       [3, 3, 4, 4, 5, 5],
       [6, 6, 7, 7, 8, 8],
       [6, 6, 7, 7, 8, 8]])

```
**make_flds**
Construct fields of a uniform dtype and column names.
```
make_flds(n=2, as_type='float', names=None, def_name="col")
Out[88]: dtype([('col_00', '<f8'), ('col_01', '<f8')])
```

**nd_struct**
Create a named array from an ndarray.  There are several other variants within the tools.py script.
```
nd_struct(a)
Out[89]: 
array([( 0,  1,  2), ( 3,  4,  5), ( 6,  7,  8), ( 9, 10, 11), (12, 13, 14)],
      dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4')])
```

**rc_vals**

Make an array of row/column/values from an ndarray.

```
a = np.arange(5*3).reshape(5,3)

rc_vals(a)

array([(0, 0,  0), (1, 0,  1), (2, 0,  2), (0, 1,  3), (1, 1,  4), (2, 1,  5), (0, 2,  6),
       (1, 2,  7), (2, 2,  8), (0, 3,  9), (1, 3, 10), (2, 3, 11), (0, 4, 12), (1, 4, 13),
       (2, 4, 14)], dtype=[('Row', '<i8'), ('Col', '<i8'), ('Val', '<i4')])

```

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
