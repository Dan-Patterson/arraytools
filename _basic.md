**_basic.py**

*Sample array*

```
a = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
              [ 9, 10, 11, 12, 13, 14, 15, 16, 17],
              [18, 19, 20, 21, 22, 23, 24, 25, 26],
              [27, 28, 29, 30, 31, 32, 33, 34, 35],
              [36, 37, 38, 39, 40, 41, 42, 43, 44],
              [45, 46, 47, 48, 49, 50, 51, 52, 53],
              [54, 55, 56, 57, 58, 59, 60, 61, 62],
              [63, 64, 65, 66, 67, 68, 69, 70, 71],
              [72, 73, 74, 75, 76, 77, 78, 79, 80]])
```

**array_info(a)**

This function returns array information, basic and other.

```
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
