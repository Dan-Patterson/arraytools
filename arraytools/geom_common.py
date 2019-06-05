# -*- coding: utf-8 -*-
"""
===========
geom_common
===========

Script :   geom_common.py

Author :   Dan_Patterson@carleton.ca

Modified : 2019-04-02

Purpose :  Common tools for working with arrays that represent geometry objects

Notes
-----
Sample array data, 2000 points with ID, X and Y::

    data = 'C:/Git_Dan/arraytools/Data/pnts_2K_id_x_y.npy'
    a = np.load(data)
    a0 = a[['Xs', 'Ys']]
    xy = _new_view_(a0)
    %timeit _new_view_(a0)
    # 1.96 µs ± 28 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import warnings
import numpy as np

v =  np.version.version.split('.')[1]  # version check
if int(v) >= 16:
    from numpy.lib.recfunctions import structured_to_unstructured as stu

warnings.simplefilter('ignore', FutureWarning)

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
def _new_view_(a):
    """View a structured array x,y coordinates as an ndarray to facilitate
    some array calculations.

    NOTE,  see _view_ for the same functionality
    """
    a = np.asanyarray(a)
    dt_len = len(a.dtype)
    if dt_len > 1:
        shp = a.shape[0]
        a = a.view(dtype='float64')
        a = a.reshape(shp, dt_len)
    return a


def _view_(a):
    """Return a view of the array using the dtype and length

    Notes
    -----
    The is a quick function.  The expectation is that the array contains a
    uniform dtype (e.g 'f8').  For example, coordinate values in the form
    ``dtype([('X', '<f8'), ('Y', '<f8')])`` maybe with a Z.

    References
    ----------
    ``structured_to_unstructured`` in np.lib.recfunctions and its imports.
    `<https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py>`_.

    """
    v =  np.version.version.split('.')[1]  # version check
    if int(v) >= 16:
        from numpy.lib.recfunctions import structured_to_unstructured as stu
        return stu(a)
    else:
        names = a.dtype.names
        z = np.zeros((a.shape[0], 2), dtype=np.float)
        z[:,0] = a[names[0]]
        z[:,1] = a[names[1]]
        return z


def _reshape_(a):
    """Reshape arrays, structured or recarrays of coordinates to a 2D ndarray.

    Returns
    -------
    The length of the dtype is checked. Only object ('O') and arrays with
    a uniform dtype return 0.  Structured and recarrays will yield 1 or more.
    Array dtypes are stripped and the array reshaped.

    >>> a = np.array([(341000., 5021000., 10), (341000., 5022000., 20.),
    ...               (342000., 5022000., 30.), (341000., 5021000., 40)],
    ...             dtype=[('X', '<f8'), ('Y', '<f8'), ('Z', '<f8')])
    |  # ---- becomes:
    >>> a = np.array([[  341000.,  5021000.], [  341000.,  5022000.],
                      [  342000.,  5022000.], [  341000.,  5021000.]])
    >>> a.dtype = dtype('float64')
    |  # ---- 3D arrays are collapsed to 2D
    >>> a.shape = (2, 5, 2) => np.product(a.shape[:-1], 2) => (10, 2)

    Object arrays are processed object by object but assumed to be of a
    common dtype within, as would be expected from a gis package.
    """
    if not isinstance(a, np.ndarray):
        try:
            a = np.asarray(a)
        except:
            raise ValueError("\nAn array is required...")
            return None
    shp = len(a.shape)
    _len = len(a.dtype)
    if a.dtype.kind == 'O':
        if len(a[0].shape) == 1:
            return np.asarray([_view_(i) for i in a])
        return a  #_view_(a)
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

