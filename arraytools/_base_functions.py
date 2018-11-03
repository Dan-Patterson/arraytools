# -*- coding: utf-8 -*-
"""
_base_functions
===============

Script :   _base_functions.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-10-27

Purpose:  tools for working with numpy arrays

Useage :

References
----------
`< >`_.
`< >`_.
---------------------------------------------------------------------
"""

import sys
from textwrap import dedent, indent
import numpy as np
#from arcpytools import fc_info, tweet  #, prn_rec, _col_format
#import arcpy

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
           'n_largest',     # (2) size-based
           'n_smallest',
           'num_to_nan',    # (3) masking]
           'num_to_mask'
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
    elif not isinstance(a, (np.ndarray, np.ma.core.MaskedArray)):
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
    :  |__shape {}\n    :  |__ndim  {}\n    :  |__size  {}
    :  |__bytes {}\n    :  |__type  {}\n    :  |__strides  {}
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
# ---- ndarray section, change format or arrangement ----
# ----------------------------------------------------------------------
# ---- (2) size-based .... n largest, n_smallest
#
def n_largest(a, num=1, by_row=True):
    """Return the 'num' largest entries in an array by row sorted by column.
    Array dimensions <=3 supported
    """
    assert a.ndim <= 3, "Only arrays with ndim <=3 supported"
    if not by_row:
        a = a.T
    num = min(num, a.shape[-1])
    if a.ndim == 1:
        b = np.sort(a)[-num:]
    elif a.ndim >= 2:
        b = np.sort(a)[..., -num:]
    else:
        return None
    return b


def n_smallest(a, num=1, by_row=True):
    """Return the 'n' smallest entries in an array by row sorted by column.
    Array dimensions <=3 supported
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


# ---- (3) masking ... num_to_nan, num_to_mask ... code section .... ----
#
def num_to_nan(a, nums=None):
    """Reverse of nan_to_num introduced in numpy 1.13

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
        return a
    else:
        m = np.isin(a, nums, assume_unique=False, invert=False)
        nums = np.array(nums)
        b = np.ma.MaskedArray(a, mask=m, hard_mask=hardmask)
    return b


# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
if len(sys.argv) == 1:
    testing = True
    # parameters here
else:
    testing = False
    # parameters here
#
if testing:
    print('\nScript source... {}'.format(script))
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """

