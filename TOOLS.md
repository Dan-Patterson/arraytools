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

**block_arr**

Similar to stride but using a jumping window rather than a moving window.

```
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])

block_arr(a, win=[2, 2], nodata=-1)
[[[ 0  1]
  [ 4  5]]

 [[ 2  3]
  [ 6  7]]

 [[ 8  9]
  [-1 -1]]

 [[10 11]
  [-1 -1]]]
```

**stride**
Stride array with a specified window size.
```
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])

stride(a, (3, 3))  # ... stride an array ....
[[[ 0  1  2]
  [ 4  5  6]
  [ 8  9 10]]

 [[ 1  2  3]
  [ 5  6  7]
  [ 9 10 11]]]
```

**change_arr**
Change a structured array including reordering and/or dropping fields.

```
b = np.array([(0, 1,  2,  3), (4, 5,  6,  7), (8, 9, 10, 11)],
             dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')])

change_arr(b, order=['B', 'C', 'A'], prn=False)  # Array 'b', reordered with 2 fields dropped...

array([[(1,  2, 0)],
       [(5,  6, 4)],
       [(9, 10, 8)]],
      dtype=[('B', '<i4'), ('C', '<i4'), ('A', '<i4')])

```
**scale**
Scale the elements of an array by a factor

```
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])

scale(a, x=2, y=2, num_z=None)

[[ 0  0  1  1  2  2  3  3]
 [ 0  0  1  1  2  2  3  3]
 [ 4  4  5  5  6  6  7  7]
 [ 4  4  5  5  6  6  7  7]
 [ 8  8  9  9 10 10 11 11]
 [ 8  8  9  9 10 10 11 11]]
```

**split_array**
```
b =np.array([(0, 1,  2,  3), (4, 5,  6,  7), (8, 9, 10, 11)],
           dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')])

split_array(b, 'A', False)
 
[
  array([(0, 1, 2, 3)], dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')]),
  array([(4, 5, 6, 7)], dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')]),
  array([(8, 9, 10, 11)], dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')])
  ]
```

**rolling_stats**
Statistics for a strided arrays
```
rolling_stats(a, no_null=True, prn=True)

min, max, mean, sum, std, var, ptp
(0, 53, 26.5, 26.5, 1431, 15.58578412100805, 242.91666666666666, 53)
```
