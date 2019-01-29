# -*- coding: utf-8 -*-
"""
_basic
======

Script :   _basic.py

Author :   Dan_Patterson@carleton.ca

Modified : 2019-01-06

Purpose:  tools for working with numpy arrays

Useage :
--------
`_base` : some functions
::
  arr_info, art_info, even_odd, n_largest, n_smallest, num_to_mask,
  num_to_nan, pad_even_odd, pad_nan, pad_zero, reshape_options, shape_to2D

References
----------

`<https://docs.python.org/3/library/itertools.html#itertools-recipes>`_.

Functions
---------
**n largest, n_smallest**

>>> a = np.arange(0, 9).reshape(3, 3)
>>> n_largest(a, num=2, by_row=True)
array([[1, 2],
       [4, 5],
       [7, 8]])
>>> n_largest(a, num=2, by_row=False)
array([[3, 4, 5],
       [6, 7, 8]])

**num_to_nan, num_to_mask** : nan stuff

>>> a = np.arange(6)  # array([0, 1, 2, 3, 4, 5])
>>> num_to_nan(a, nums=[2, 3])
array([ 0.,  1., nan, nan,  4.,  5.])
>>> num_to_mask(a, nums=[2, 3]) ...
masked_array(data = [0 1 - - 4 5],
             mask = [False False  True  True False False],
       fill_value = 999999)

---------------------------------------------------------------------
"""
# pylint: disable=C0103
# pylint: disable=R1710
# pylint: disable=R0914

import sys
from textwrap import dedent, indent
import numpy as np
#from arcpytools import fc_info, tweet  #, prn_rec, _col_format
#import arcpy

epsilon = sys.float_info.epsilon  # note! for checking


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


#import numpy.core.numerictypes as ntypes

type_keys = np.typecodes.keys()
type_vals = np.typecodes.values()


__all__ = ['arr_info',      # (1) info functions
           'keep_ascii',    # (2) chararray
           'is_float',
           'keep_nums',
           'del_punc',
           'n_largest',     # (3) ndarray... size-based
           'n_smallest',
           'num_to_nan',    # (4) masking
           'num_to_mask',
           'even_odd',      # (5) padding, diagonal
           'pad_even_odd',
           'pad_nan',
           'pad_zero',
           'fill_diagonal',
           'shape_to2D',    # (6) reshaping arrays
           'reshape_options',
           'cartesian',     # (7) form from-to pairs from one or two arrays
           'all_pairs',
           'ft_pairs'
           ]

# ----------------------------------------------------------------------
# ---- (1) info .... code section ----
def arr_info(a=None, prn=True):
    """Returns basic information about an numpy array.

    Requires:
    --------
    a : array
        An array to return basic information on.
    prn : Boolean
        True to print, False to return as string.

    Returns
    -------
    Example array information.

    >>> a = np.arange(2. * 3.).reshape(2, 3) # quick float64 array
    >>> arr_info(a)
        Array information....
         OWNDATA: if 'False', data are a view
        flags....
        ... snip ...
        array
            |__shape (2, 3)
            |__ndim  2
            |__size  6
            |__bytes
            |__type  <class 'numpy.ndarray'>
            |__strides  (24, 8)
        dtype      float64
            |__kind  f
            |__char  d
            |__num   12
            |__type  <class 'numpy.float64'>
            |__name  float64
            |__shape ()
            |__description
                 |__name, itemsize
                 |__['', '<f8']
    ---------------------
    """
    if a is None:
        print(arr_info.__doc__)
        return None
    if not isinstance(a, (np.ndarray, np.ma.core.MaskedArray)):
        s = "\n... Requires a numpy ndarray or variant...\n... Read the docs\n"
        print(s)
        return None
    frmt = """
    :---------------------
    :Array information....
    : OWNDATA: if 'False', data are a view
    :flags....
    {}
    :array
    :  |__shape {}\n    :  |__ndim  {}\n    :  |__size  {:,}
    :  |__bytes {:,}\n    :  |__type  {}\n    :  |__strides  {}
    :dtype      {}
    :  |__kind  {}\n    :  |__char  {}\n    :  |__num   {}
    :  |__type  {}\n    :  |__name  {}\n    :  |__shape {}
    :  |__description
    :  |  |__name, itemsize"""
    dt = a.dtype
    flg = indent(a.flags.__str__(), prefix=':   ')
    info_ = [flg, a.shape, a.ndim, a.size,
             a.nbytes, type(a), a.strides, dt,
             dt.kind, dt.char, dt.num, dt.type, dt.name, dt.shape]
    flds = sorted([[k, v] for k, v in dt.descr])
    out = dedent(frmt).format(*info_) + "\n"
    leader = "".join([":     |__{}\n".format(i) for i in flds])
    leader = leader + ":---------------------"
    out = out + leader
    if prn:
        print(out)
    else:
        return out


# ----------------------------------------------------------------------
# ---- (2) chararray section ----
# ----------------------------------------------------------------------
def keep_ascii(s):
    """Remove non-ascii characters which may be bytes or unicode characters
    """
    if isinstance(s, bytes):
        u = s.decode("utf-8")
        u = "".join([['_', i][ord(i) < 128] for i in u])
        return u
    return s


def is_float(a):
    """float check"""
    try:
        np.asarray(a, np.float_)
        return True
    except ValueError:
        return False


def keep_nums(s):
    """Remove all non-numbers and return an integer.
    """
    s = keep_ascii(s)
    s = "".join([i for i in s if i.isdigit() or i == " "]).strip()
    return int(s)


def del_punc(s, keep_under=False, keep_space=False):
    """Remove punctuation with options to keep the underscore and spaces.
    If keep_space is True, then they will not be replaced with an underscore.
    False, will replace them.  Check for bytes as well.
    """
    s = keep_ascii(s)
    repl = ' '
    punc = list('!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~')
    if keep_under:
        punc.append('_')
    if not keep_space:
        punc.append(' ')
        repl = ''
    s = "".join([[i, repl][i in punc] for i in s])
    return s


def del_punc_space(name, repl_with='_'):
    """delete punctuation and spaces and replace with '_'"""
    punc = list('!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~ ')
    return "".join([[i, repl_with][i in punc] for i in name])


# ----------------------------------------------------------------------
# ---- ndarray section, change format or arrangement ----
# ----------------------------------------------------------------------
# ---- (3) size-based .... n largest, n_smallest
#
def n_largest(a, num=1, by_row=True):
    """Return the`num` largest entries in an array by row sorted by column, or
    by column sorted by row.

    Parameters:
    -----------
    a : ndarray
        Array dimensions <=3 supported
    num : integer
        The number of elements to return
    by_row : boolean
        True for returns by row, False to determine by column
    """
    assert a.ndim <= 3, 'Only arrays with ndim <=3 supported'
    if not by_row:
        a = a.T
    num = min(num, a.shape[-1])
    if a.ndim == 1:
        b = np.sort(a)[-num:]
    elif a.ndim >= 2:
        b = np.sort(a)[..., -num:]
    else:
        return None
    if not by_row:
        b = b.T
    return b


def n_smallest(a, num=1, by_row=True):
    """Return the 'n' smallest entries in an array by row sorted by column.
    see `n_largest` for parameter description
    """
    assert a.ndim <= 3, "Only arrays with ndim <=3 supported"
    if not by_row:
        a = a.T
    num = min(num, a.shape[-1])
    if a.ndim == 1:
        b = np.sort(a)[:num]
    elif a.ndim >= 2:
        b = np.sort(a)[..., :num]
    else:
        return None
    return b


# ---- (4) masking ... num_to_nan, num_to_mask ... code section .... ----
#
def num_to_nan(a, nums=None):
    """Reverse of np.nan_to_num introduced in numpy 1.13

    Example
    -------
    >>> a = np.arange(10)
    >>> num_to_nan(a, num=[2, 3])
    array([  0.,   1.,   nan,  nan,   4.,   5.,   6.,   7.,   8.,   9.])
    """
    a = a.astype('float64')
    if nums is None:
        return a
    if isinstance(nums, (list, tuple, np.ndarray)):
        m = np.isin(a, nums, assume_unique=False, invert=False)
        a[m] = np.nan
    else:
        a = np.where(a == nums, np.nan, a)
    return a


def num_to_mask(a, nums=None, hardmask=True):
    """Reverse of nan_to_num introduced in numpy 1.13

    Example
    -------
    >>> a = np.arange(10)
    >>> art.num_to_mask(a, nums=[1, 2, 4])
    masked_array(data = [0 - - 3 - 5 6 7 8 9],
                mask = [False  True  True False  True False
                        False False False False], fill_value = 999999)
    """
    if nums is None:
        ret = a
    else:
        m = np.isin(a, nums, assume_unique=False, invert=False)
        nums = np.array(nums)
        ret = np.ma.MaskedArray(a, mask=m, hard_mask=hardmask)
    return ret


# ---- (5) padding arrays  ... even_odd, pad_even_odd, pad_nan, pad_zero
#
def even_odd(a):
    """Even/odd from modulus.  Returns 0 for even, 1 for odd
    """
    prod = np.cumprod(a.shape)[0]
    return np.mod(prod, 2)


def pad_even_odd(a):
    """To use when padding a strided array for window construction

    See also: art.num_to_nan, art.pad_nan, art.pan_zero
    """
    p = even_odd(a)
    ap = np.pad(a, pad_width=(1, p), mode='constant', constant_values=(0, 0))
    return ap


def pad_nan(a, nan_edge=True):
    """Pad a sliding array to allow for stats, padding uses np.nan

    See also: art.num_to_nan, art.pan_zero, pad_even_odd
    """
    a = a.astype('float64')
    if nan_edge:
        cv = (np.NaN, np.NaN)
        a = np.pad(a, pad_width=(1, 1), mode='constant', constant_values=cv)
    return a


def pad_zero(a, n=1):
    """To use when padding a strided array for window construction. n = number
    of zeros to pad arround the array

    See also: art.num_to_nan, art.pad_nan, pad_even_odd
    """
    ap = np.pad(a, pad_width=(n, n), mode='constant', constant_values=(0, 0))
    return ap


def fill_diagonal(n=5, seq=None):
    """Fill the diagonal of a square array with a sequence

    Parameters:
    -----------
    n : integer
        Rows and columns to create.
    seq : array-like
        Sequence to put on the diagonal.  It need not but sequential
    """
    aye = np.eye(n)
    row_col = np.diag_indices_from(aye)
    if seq is None:
        seq = np.arange(n)
    elif len(seq) > n:
        seq = seq[:n]
    elif len(seq) < n:
        seq = np.concatenate((np.array(seq), np.zeros(n)))[:n]
    aye[row_col] = seq
    return aye


def strip_whitespace(a):
    """Strip unicode whitespace from an array.  Remove the first backslash::

    '\\t', '\\n', '\\x0b', '\\x0c', '\\r', '\\x1c', '\\x1d', '\\x1e','\\x1f',
    ' ', '\\x85', '\\xa0', '\\u1680', '\\u2000', '\\u2001', '\\u2002',
    '\\u2003', '\\u2004', '\\u2005', '\\u2006', '\\u2007', '\\u2008',
    '\\u2009', '\\u200a', '\\u2028', '\\u2029', '\\u202f', '\\u205f', '\\u3000'
    >>> others = ['\\u200B', '\\u200C', '\\u200D', '\\u2060', '\\uFEFF']

    References:
    -----------
    `<https://en.wikipedia.org/wiki/Whitespace_character>`_.

    """
    assert a.dtype.kind in ('U', 'S'), "Only numeric arrays supported"
    return np.char.strip(a)


# ---- (6) reshaping arrays ... shape_to2D, reshape_options
#
def shape_to2D(a, stack2D=True, by_column=False):
    """Reshape an ndim array to a 2D array for print formatting and other uses.

    a : array
        array ndim >= 3
    stack2D : boolean
      True, swaps, then stacks the last two dimensions row-wise.

      False, the first two dimensions are raveled then vertically stacked

    >>> shp = (2, 2, 3)
    >>> a = np.arange(np.prod(shp)).reshape(shp)
    array([[[ 0,  1,  2],
            [ 3,  4,  5]],
           [[ 6,  7,  8],
            [ 9, 10, 11]]])

    2D stack by row

    >>> re_shape(a, stack2D=True,  by_column=False)  # np.hstack((a[0], a[1]))
    array([[ 0,  1,  2,  6,  7,  8],
           [ 3,  4,  5,  9, 10, 11]])

    2D stack raveled by row

    >>> re_shape(a, stack2D=False, by_column=False)
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11]])

    2D stack by column

    >>> re_shape(a, True, True)
    array([[ 0,  3],
           [ 1,  4],
           [ 2,  5],  # note here a[0] is translated and stacked onto a[1]
           [ 6,  9],
           [ 7, 10],
           [ 8, 11]])

    >>>  re_shape(a, False, True)
    array([[ 0,  6],
           [ 1,  7],
           [ 2,  8],  # note here a[0] becomes raveled and translated and
           [ 3,  9],  # a[1] is stacked column-wise to it.
           [ 4, 10],
           [ 5, 11]])

    For other shapes::

        shp          re_shape(a).shape
        (3, 4)       (4, 3)
        (2, 3, 4)    (3, 8)
        (2, 3, 4, 5) (3, 40)

    np.transpose and np.swapaxes are related

    >>> np.all(np.swapaxes(a, 0, 1) == np.transpose(a, (1, 0, 2)))
    """
    shp = a.shape
    if stack2D:
        out = np.swapaxes(a, 0, 1).reshape(shp[1], np.prod((shp[0],) + shp[2:]))
    else:
        m = 2
        n = len(shp) - m
        out = a.reshape(np.prod(shp[:n], dtype='int'), np.prod(shp[-m:]))
    if by_column:
        out = out.T
    return out


def reshape_options(a):
    """Alternative shapes for a numpy array.

    Parameters:
    -----------
    a : ndarray
        The ndarray with ndim >= 2

    Returns:
    --------
    An object array containing the shapes of equal or lower dimension,
    excluding ndim=1

    >>> a.shape # => (3, 2, 4)
    array([(2, 12), (3, 8), (4, 6), (6, 4), (8, 3), (12, 2), (2, 3, 4),
           (2, 4, 3), (3, 2, 4), (3, 4, 2), (4, 2, 3), (4, 3, 2)],
          dtype=object)

    Notes:
    ------
    >>> s = list(a.shape)
    >>> case = np.array(list(chain.from_iterable(permutations(s, r)
                        for r in range(len(s)+1)))[1:])
    >>> prod = [np.prod(i) for i in case]
    >>> match = np.where(prod == size)[0]

    References:
    -----------
    `<https://docs.python.org/3/library/itertools.html#itertools-recipes>`_.

    """
    from itertools import permutations, chain
    s = list(a.shape)
    n = len(s) + 1
    ch = list(chain.from_iterable(permutations(s, r) for r in range(n)))
    case0 = np.array(ch[1:])
    case1 = [i + (-1,) for i in case0]
    new_shps = [a.reshape(i).shape for i in case1]
    z = [i[::-1] for i in new_shps]
    new_shps = new_shps + z
    new_shps = [i for i in np.unique(new_shps) if 1 not in i]
    new_shps = np.array(sorted(new_shps, key=len, reverse=False))
    return new_shps


# ---- (7) form from-to pairs from one or two arrays
#
def cartesian(arrays):
    """Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1D or 2D arrays to form the cartesian product of.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6]))
    >>> cartesian((['a', 'b'], ['a', 'b', 'c']))
    array([[1, 4, 6],     array([['a', 'a'],
           [1, 5, 6],            ['a', 'b']
           [2, 4, 6],            ['a', 'c'],
           [2, 5, 6],            ['b', 'a'],
           [3, 4, 6],            ['b', 'b'],
           [3, 5, 6]])           ['b', 'c']], dtype='<U1')

    References:
    -----------
    `<https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-
    x-and-y-array-points-into-single-array-of-2d-points>`_.

    René Descartes : `<https://en.wikipedia.org/wiki/René_Descartes>`_.
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = [len(x) for x in arrays]
    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T
    out = []
    for n, arr in enumerate(arrays):
        sub = arrays[n][ix[:, n]]
        if sub.ndim == 1:
            out.append(sub.reshape(sub.shape[0],1))
        else:
            out.append(sub)
    return np.hstack(out) #, out


def all_pairs(a, to_3d=False):
    """Make pairs for all combinations in a 2D array.  It is a stacked array
    version of `ft_pairs`. More combinations without duplicates are returned

    Parameters:
    -----------
    a : array-like
        A (N,2) array or list such as pairs of x,y coordinates.
    to_3d
        If True, then from-to pairs are created within a 3D structure.
        If False, than a 2D array is created with each row representing a
        fromX, fromY, toX, toY sequence

    References:
    -----------
    `<https://stackoverflow.com/questions/46339926/finding-all-combinations
    -of-paired-numpy-arrays-via-meshgrid>`_.

    Example:
    --------

    >>> a = np.arange(6).reshape(3,2)
    >>> b = np.arange(7, 11).reshape(2,2)
    >>> ab = np.concatenate((a, b), axis=0)
    ab
    array([[ 0,  1],
           [ 2,  3],
           [ 4,  5],
           [ 7,  8],
           [ 9, 10]])
    >>> all_pairs(ab)
    array([[ 0,  1,  2,  3],   ** within `a` pair
           [ 0,  1,  4,  5],   ** within `a` pair
           [ 0,  1,  7,  8],
           [ 0,  1,  9, 10],
           [ 2,  3,  4,  5],
           [ 2,  3,  7,  8],
           [ 2,  3,  9, 10],
           [ 4,  5,  7,  8],
           [ 4,  5,  9, 10],
           [ 7,  8,  9, 10]])  ** within `b` pair

    See also:
    ---------
    1. `ft_pairs` produces similar results but not the within group combinations
       marked with ** above

    2. `cartesian` one of the standard approaches
    """
    if a.ndim != 2:
        return "(N,2) array required"
    a = np.asarray(a)
    r, c = np.triu_indices(len(a), 1)
    if to_3d:
        return np.stack((a[r], a[c]), axis=1)
    return np.hstack((a[r], a[c]))


def ft_pairs(a, b, to_2D=True):
    """Make pairs or coordinates from arrays of the same or different lengths.
    The current implementation is designed to pair coordinate data.

    Parameters:
    -----------
    a : array like
        Two (N, 2) arrays or lists such pairs of x,y coordinates.
    to_2d : boolean
        If True, then (from, to pairs) are created within a 3D structure.

        If False, than a 2D array is created with each row representing a
        fromX, fromY, toX, toY sequence

    References:
    -----------
    A couple

    `<https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-
    x-and-y-array-points-into-single-array-of-2d-points>`_.

    `<https://stackoverflow.com/questions/34502254/vectorizing-haversine-
    distance-calculation-in-python>`_.

    Example:
    --------

    >>> a = np.arange(6).reshape(3,2)
    >>> b = np.arange(7, 11).reshape(2,2)
         array([[0, 1],   array([[ 7,  8],
                [2, 3],          [ 9, 10]])
                [4, 5]])
    >>> z0 = make_pairs(a, b, False)
    >>> prn(z0)
    array([[ 0,  1,  7,  8],
           [ 0,  1,  9, 10],
           [ 2,  3,  7,  8],
           [ 2,  3,  9, 10],
           [ 4,  5,  7,  8],
           [ 4,  5,  9, 10]])
    >>> np.asarray([[*i, *j] for i in a for j in b]  # same as
    """
    if (a.ndim!=b.ndim) and (a.ndim != 2):
        return "(N,2) arrays required"
    out = []
    for _, j in enumerate(a):
        for _, l in enumerate(b):
            out.append((*j, *l))
    if to_2D:
        out = np.asarray(out)
    return out

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    # print the script source name.
    print("Script... {}".format(script))
