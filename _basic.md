**_basic.py**

Some examples of functions in the \_basic portion of arraytools.

**array_info**

This function returns array information, basic and other.

```
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
              
array_info(a)

:---------------------
:Array information....
: OWNDATA: if 'False', data are a view
:flags....
:     C_CONTIGUOUS : True
:     F_CONTIGUOUS : False
:     OWNDATA : False
:     WRITEABLE : True
:     ALIGNED : True
:     WRITEBACKIFCOPY : False
:     UPDATEIFCOPY : False
:array
:  |__shape (9, 9)
:  |__ndim  2
:  |__size  81
:  |__bytes 324
:  |__type  <class 'numpy.ndarray'>
:  |__strides  (36, 4)
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
```

**is_finite**

Simply runs a np.isfinite check on an array, with the option to return elements that are finite.

```
a0 = np.array([1., np.NAN, np.inf, np.NINF, 5])  # np.nan, np.NaN, np.inf etc

is_finite(a0, return_finite=True)  # optional return of finite values

(False, array([1., 5.]))           # overall, the array contains elements which are not finite
```

**n_largest and n_smallest**

```
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])

n_largest(a, num=1, by_row=True)

array([[ 3],
       [ 7],
       [11]])

n_largest(a, num=1, by_row=False)

array([[ 8,  9, 10, 11]])

```

**num_to_nan and num_to_mask**

```
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])

num_to_nan(a, nums=[1,6,9])

array([[ 0., nan,  2.,  3.],
       [ 4.,  5., nan,  7.],
       [ 8., nan, 10., 11.]])

num_to_mask(a, nums=[1,6,9])

masked_array(
  data=[[0, -, 2, 3],
        [4, 5, -, 7],
        [8, -, 10, 11]],
  mask=[[False,  True, False, False],
        [False, False,  True, False],
        [False,  True, False, False]],
  fill_value=999999)
```

**pad_even_odd, pad_nan and pad_zero**

```
pad_even_odd(a)
 
array([[ 0,  0,  0,  0,  0,  0],
       [ 0,  0,  1,  2,  3,  0],
       [ 0,  4,  5,  6,  7,  0],
       [ 0,  8,  9, 10, 11,  0],
       [ 0,  0,  0,  0,  0,  0]])

pad_nan(a, nan_edge=True)

array([[nan, nan, nan, nan, nan, nan],
       [nan,  0.,  1.,  2.,  3., nan],
       [nan,  4.,  5.,  6.,  7., nan],
       [nan,  8.,  9., 10., 11., nan],
       [nan, nan, nan, nan, nan, nan]])

pad_zero(a, n=1)

array([[ 0,  0,  0,  0,  0,  0],
       [ 0,  0,  1,  2,  3,  0],
       [ 0,  4,  5,  6,  7,  0],
       [ 0,  8,  9, 10, 11,  0],
       [ 0,  0,  0,  0,  0,  0]])
```
