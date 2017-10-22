# -*- coding: UTF-8 -*-
"""
:Script:   flds.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2016-10-15
:Purpose:  To provide the capabilities to reorder fields in a structured
: or recarray.
:
:Notes  ---- some examples for ndarray and structured/recarray
:
: ---- input array ----
:  a = np.arange(4*5).reshape((4,5))
:  a  # ndarray all same data type
:  array([[ 0,  1,  2,  3,  4],
:         [ 5,  6,  7,  8,  9],
:         [10, 11, 12, 13, 14],
:         [15, 16, 17, 18, 19]])
:
: ---- reorder columns ----
:  b = a[:, [2,1,0,3,4]]   # reorder the columns, keeping the rows
:  c = a[:, [0,2,3]]       # delete columns 1 and 4
:  d = a[[0,1,3,4], :]     # delete row 2, keeping the columns
:  e = a[[0,1,3], [1,2,3]] # keep [0,1], [1,2], [3,3] => ([ 1,  7, 18])
:
: ---- ndarray to structured array ----
: -- keep the dtype the same
:  aa = nd_struct(a)       # produce a structured array from inputs
:  aa.reshape(-1,1)   # structured array
:  array([[(0, 1, 2, 3, 4)],
:         [(5, 6, 7, 8, 9)],
:         [(10, 11, 12, 13, 14)],
:         [(15, 16, 17, 18, 19)]],
:     dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'),
:            ('D', '<i4'), ('E', '<i4')])
:
: -- upcast the dtype
:  a_f = nd_struct(a.astype('float'))  # note astype allows a view
:  a_f
:  array([(0.0, 1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0, 9.0),
:         (10.0, 11.0, 12.0, 13.0, 14.0), (15.0, 16.0, 17.0, 18.0,  19.0)],
:    dtype=[('A', '<f8'), ('B', '<f8'), ('C', '<f8'),
:           ('D', '<f8'), ('E', '<f8')])
:
: ---- reorder the input fields ----
: aa[['B','D','A','E','C']].reshape(-1,1)
: array([[(1, 3, 0, 4, 2)],
:        [(6, 8, 5, 9, 7)],
:        [(11, 13, 10, 14, 12)],
:        [(16, 18, 15, 19, 17)],
:        [(21, 23, 20, 24, 22)],
:        [(26, 28, 25, 29, 27)]],
:       dtype=[('B', '<i4'), ('D', '<i4'), ('A', '<i4'),
;              ('E', '<i4'), ('C', '<i4')])
:
: ---- slice both columns and rows ----
: cc = aa[['A','E']][[1,3,5]]
: cc.reshape(-1,1)
: array([[(5, 9)],
:        [(15, 19)],
:        [(25, 29)]],
:       dtype=[('A', '<i4'), ('E', '<i4')])
:
:References
:
"""
import sys
import numpy as np
from numpy.lib import recfunctions as rfn
from textwrap import dedent
np.set_printoptions(edgeitems=4, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter={'float': '{: 0.3f}'.format})
# Variables
script = sys.argv[0]


# Functions
def fld_info(a, verbose=True):
    """Return field info for an array.  This may be required before
    : one can use the reorder_flds since the field names are needed for
    : structured or recarrays.
    :Requires:
    :--------
    : an array of any type
    :Returns:
    :-------
    : field/column names, array shape and dtype
    :
    """
    shp = a.shape
    dt = a.dtype.descr
    names = a.dtype.names
    if verbose:
        frmt = """
        Array info...
        names  {}
        shape  {}
        dtype  {}
        """
        print(dedent(frmt).format(names, shp, dt))
    return names, dt, shp


def nd_struct(a):
    """ convert ndarray to structured/recarray
    :Requires
    :--------
    : a - ndarray with a uniform dtype, the field names are assigned
    :     from an alphabetical list up to 52 fields
    :Returns
    :-------
    : array with reordered fields and/or sliced by row or column as specified
    :
    """
    if a.ndim != 2:
        print("A 2D ndarray is required")
        return a
    rows, cols = a.shape
    dt_name = a.dtype.name
    fld_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")[:cols]
    dt = [(i, dt_name) for i in fld_names]
    aa = np.zeros((rows,), dtype=dt)
    names = aa.dtype.names
    for i in range(a.shape[1]):
        aa[names[i]] = a[:, i]
    return aa


def alter_flds(a, order=[]):
    """Reorder and/or drop columns in an array using slicing.  Fields not
    :  included will be dropped in the output array. see help(alter_flds).
    :
    :  To remove fields, simply leave them out of the list.  The order of the
    :  remaining fields will be reflected in the output array.
    :
    :  This is a convenience function.... see the module header for one-liner
    :  syntax.
    :
    : use: fld_info(a,verbose=True)
    :    provides field names which can be copied for use here.
    :
    :Requires
    :--------
    :  If using a structured or recarray, the desired field order is required.
    :  An ndarray without named fields, will require the numerical order of the
    :  fields.
    """
    names, dt, shp = fld_info(a, verbose=False)
    if names is None:
        b = a[:, order]
    else:
        out_flds = []
        out_flds = [i for i in order if i in names]
        missing = [i for i in names if i not in order]
        missing.extend([i for i in order if i not in out_flds])
        frmt = "Alter Fields...,\nField(s) not found or missing ...\n{}"
        print(frmt.format(missing))
        b = a[out_flds]
    return b


def _demo():
    """Run some demonstrations of the functions"""
    a0 = np.arange(50).reshape((10, 5))
    order = [2, 0, 4]
    a1 = alter_flds(a0, order)
    frmt = """
    (1) Input numeric array...
    {!r:}\n
    (2) Alter field order for ndarray output...order={}
    {!r:}
    """
    print(dedent(frmt).format(a0, order, a1))
    #
    b0 = nd_struct(a0)
    dt1 = [('A', '<i8'), ('B', '<i8'), ('C', '<f8'),
           ('D', '<i8'), ('E', '<f8')]
    b1 = b0.astype(dt1)
    print("\n(3) Input structured array...\n{!r:}".format(b1.reshape(-1, 1)))
    print("\n(4) Alter fields for structured array ")
    names, dt, shp = fld_info(b1, verbose=True)
    order = ['A', 'B', 'D', 'd', 'C']
    b2 = alter_flds(b1, order)
    frmt = "\nStructured array output...order {}\n{!r:}"
    print(frmt.format(order, b2.reshape(-1, 1)))
    return a1, b1, b2


if __name__ == "__main__":
    """   """
    a1, b1, b2 = _demo()
