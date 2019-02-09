# -*- coding: utf-8 -*-
"""
===========
geom_common
===========

Script :   geom_common.py

Author :   Dan_Patterson@carleton.ca

Modified : 2019-02-08

Purpose :  Common tools for working with arrays that represent geometry objects

Notes

References

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import numpy as np


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['_new_view_',
           '_view_',
           '_reshape_'
           ]
# ===========================================================================
# ---- def section: def code blocks go here ---------------------------------
# ---- _view and _reshape_ are helper functions -----------------------------
#
# ---- _view and _reshape_ are helper functions -----------------------------
#
def _new_view_(a):
    """View a structured array x,y coordinates as an ndarray to facilitate
    some array calculations.

    NOTE,  see _view_ for the same functionality

    """
    a = np.asanyarray(a)
    if len(a.dtype) > 1:
        shp = a.shape[0]
        a = a.view(dtype='float64')
        a = a.reshape(shp, 2)
    return a


def _view_(a):
    """Return a view of the array using the dtype and length

    Notes
    -----
    The is a quick function.  The expectation is that they are coordinate
    values in the form  dtype([('X', '<f8'), ('Y', '<f8')])
    """
    return a.view((a.dtype[0], len(a.dtype.names)))


def _reshape_(a):
    """Reshape arrays, structured or recarrays of coordinates to a 2D ndarray.

    Returns
    -------
    The length of the dtype is checked. Only object ('O') and arrays with
#    a uniform dtype return 0.  Structured, recarrays will yield 1 or more.
#    Array dtypes are stripped and the array reshaped.
#
#    >>> a = np.array([(341000., 5021000.), (341000., 5022000.),
#    ...               (342000., 5022000.), (341000., 5021000.)],
#    ...              dtype=[('X', '<f8'), ('Y', '<f8')])
#
#    becomes:
#
#    >>> a = np.array([[  341000.,  5021000.], [  341000.,  5022000.],
#                      [  342000.,  5022000.], [  341000.,  5021000.]])
#    >>> a.dtype = dtype('float64')
#
#    3D arrays are collapsed to 2D
#
#    >>> a.shape = (2, 5, 2) => np.product(a.shape[:-1], 2) => (10, 2)

    Object arrays are processed object by object but assumed to be of a
    common dtype within, as would be expected from a gis package.

    """
    if not isinstance(a, np.ndarray):
        raise ValueError("\nAn array is required...")
    shp = len(a.shape)
    _len = len(a.dtype)
    if a.dtype.kind == 'O':
        if len(a[0].shape) == 1:
            return np.asarray([_view_(i) for i in a])
        return _view_(a)
    #
    if _len == 0:
        if shp <= 2:
            return a
        if shp > 2:
            tmp = a.reshape(np.product(a.shape[:-1]), 2)
            return tmp.view(a.dtype)
    if _len == 1:
        fld_name = a.dtype.names[0]  # assumes 'Shape' field is the geometry
        return a[fld_name]
    if _len >= 2:
        if shp == 1:
            if len(a) == a.shape[0]:
                view = _view_(a)
            else:
                view = np.asanyarray([_view_(i) for i in a])
            return view
        return np.asanyarray([_view_(i) for i in a])
    return a


# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    #msg = _demo_()

