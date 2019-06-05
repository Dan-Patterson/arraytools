# -*- coding: utf-8 -*-
"""
====
saws
====

Script :   saws.py

Author :   Dan_Patterson@carleton.ca

Modified : 2019-02-14

Purpose :
    Tools for sorting arrays, arranging, weaving and shaping
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=C0325  # Unnecessary parens after...
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


__all__ = ['sort_by_columns',               # column and row sorting
           'sort_on_column',
           'sort_cols_by_row',
           'radial_sort',
           'dist_bearing_sort',
           'sort_structured',
           'sort_points',
           'view_sort',
           'xy_sort',
           'weave'
           ]


# ---- (0) helpers from arraytools.geom -------------------------------------
#
def _center_(a, remove_dup=True):
    """Return the center of an array. If the array represents a polygon, then
    a check is made for the duplicate first and last point to remove one.
    """
    if a.dtype.kind in ('V', 'O'):
        a = _new_view_(a)
    if remove_dup:
        if np.all(a[0] == a[-1]):
            a = a[:-1]
    return a.mean(axis=0)


def _new_view_(a):
    """View a structured array x,y coordinates as an ndarray to facilitate
    some array calculations.

    See `_view_` for the same functionality
    """
    a = np.asanyarray(a)
    if len(a.dtype) > 1:
        shp = a.shape[0]
        a = a.view(dtype='float64')
        a = a.reshape(shp, 2)
    return a


# ---- (1) ndarray, column and row sorting .... code section ---------------
# ----
def sort_by_columns(a, order=None, descending=False):
    """Sort a Nxm array by a column order, numbered from 0 for the first column

    Parameters
    ----------
    a : array-like
        Specifically, the array must of the same dtype with at least 2D in
        shape
    order : None, list, tuple, np.ndarray
        See the order options within the code.
        - None :  reverses shape[1]  shape = (N, 3) yields order = (2, 1, 0)
        - integer : sorts on that column
        - tuple/list : sorts on the reversed order
    >>> a = np.array([[0, 0, 0],
                      [2, 1, 1],
                      [1, 1, 2],
                      [2, 1, 2],
                      [0, 2, 2],
                      [0, 2, 0]])
    """
    a_view = a.view(a.dtype.descr * a.shape[1])
    names = a_view.dtype.names
    if order is None:
        key = names
    elif isinstance(order, int):
        key = names[order]
    elif isinstance(order, (list, tuple, np.ndarray)):
        order = order[::-1]                # reverse the order of precedence
        key = [names[i] for i in order]  # pull the name from the list
    else:
        print("Order, must be None, list, tuple or ndarray")
        return a
    idx = np.argsort(a_view, axis=0, order=key).ravel()
    a_ordered = np.ascontiguousarray(a[idx])
    if descending:
        return a_ordered[::-1]
    return a_ordered


def sort_on_column(a, col=0, descending=False):
    """Sort on a single column.

    >>> sort_rows_by_col(a, 0, True)
    >>> a =array([[0, 1, 2],    array([[6, 7, 8],
                  [3, 4, 5],           [3, 4, 5],
                  [6, 7, 8]])          [0, 1, 2]])

    References
    ----------
    Steve Tjoa on
    `<https://stackoverflow.com/questions/2828059/sorting-arrays-
    in-numpy-by-column/2828121#2828121>`_.
    """
    a = np.asarray(a)
    shp = a.shape[0]
    if not (0 <= abs(col) <= shp):
        raise ValueError("column ({}) in range (0 to {})".format(col, shp))
    a_s = a[a[:, col].argsort()]
    if descending:
        a_s = a_s[::-1]
    return a_s


def sort_cols_by_row(a, descending=False):
    """Sort the rows of an array in the order of their column values.
    Uses lexsort """
    ret = a[np.lexsort(np.transpose(a)[::-1])]
    if descending:
        ret = ret[::-1]
    return ret


def radial_sort(pnts, cent=None, distance=True, as_azimuth=False):
    """Sort about the point cloud center or from a given point

    Parameters
    ----------
    pnts : points
        An array of points (x,y) as array or list
    cent : array-like
        - A list, tuple, array of the center's x,y coordinates.
        - None, the center's coordinate is calculated from the values with
          duplicates removed

    >>> cent = [0, 0] or np.array([0, 0])

    Returns
    -------
    - The angles in the range -180, 180 x-axis oriented
    - The pnts are NOT returned sorted, you will have to use the sort_order to
      complete the sorting.

    >>> pnts_sorted = pnts[sort_order]
    """
    def _e_2d_(p, a):
        """mini e_dist for 2d points array, a, and a single point, p.
        """
        diff = a - p[np.newaxis, :]  # a and p are ndarrays
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))
    #
    pnts = pnts.view((pnts.dtype[0], len(pnts.dtype.names)))
    #pnts = _new_view_(pnts)
    if cent is None:
        cent = _center_(pnts, remove_dup=False)
    ba = pnts - cent
    ang_ab = np.arctan2(ba[:, 1], ba[:, 0])
    ang_ab = np.degrees(ang_ab)
    sort_order = np.argsort(ang_ab)
    if as_azimuth:
        ang_ab = np.where(ang_ab > 90, 450.0 - ang_ab, 90.0 - ang_ab)
    if distance:
        dist = _e_2d_(np.array([0., 0.]), ba)  # point and array
        return ang_ab, sort_order, dist
    return ang_ab, sort_order


def dist_bearing_sort(pnts, cent=None, as_struct=False):
    """Sorts points based on a radial sort of a point set (X, Y) relative to
    the center of the points.  The sort is done using the bearing, then the
    distance of a point to the center.

    Parameters
    ----------
    pnts : array
        Points as x,y pairs
    cent : array or None
        If None, the center will be calculated
    as_struct : boolean
        True, returns a structured array

    >>> xy = a0[0][['X', 'Y']]
    >>> pnts = _view_(xy) or....
    >>> s = dist_bearing_sort(a0[0][['X', 'Y']])
    | s.shape # (69688, 2)
    | 11.4 ms ± 272 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    def _xy_struct_(a):
        """construct a structured array from x,y data in an N-2 array assuming
        X is first column and Y follows
        """
        shp = a.shape
        dt = [('X', '<f8'), ('Y', '<f8'), ('angle', '<f8'), ('dist', '<f8')]
        z = np.zeros((shp[0],), dtype=dt)
        z['X'] = a[:, 0]
        z['Y'] = a[:, 1]
        z['angle'] = a[:, 2]
        z['dist'] = a[:, 3]
        return z
    #
    ang, order, dist = radial_sort(pnts, cent=cent, distance=True)
    s = np.stack((pnts[:, 0], pnts[:, 1], ang, dist), axis=1)
    s = s[order]
    h = np.arange(-180, 180.5, 0.5, dtype='float')
    hist = np.histogram(s[:, 3], bins=h)  # on distance
    cnts, low = hist
    out = []
    for i in range(1, len(low)):
        s0 = s[np.logical_and(s[:, 2] > low[i-1], s[:, 2] <= low[i])]
        d_max = s0[:, 3].max()
        delta = d_max*0.1
        w = s0[np.logical_and(s0[:, 3] > d_max-delta, s0[:, 3] <= d_max)]
        if len(w) < 5:
            w = s0
        out.append(w)
    out = np.vstack((out))
    if as_struct:
        out = _xy_struct_(out)
    return s, cnts, low, out


def sort_structured(a, sort_fields=None):
    """Sort a structured array of based on field order.

    Parameters
    ----------
    a : structured/record array
        The array requires fields identifing the equivalent of an X and Y field
    xy_fields : list
        The field names representing the fields to sort on. eg. ['X', 'Y']
    struct : boolean
        See ``sort_points_radial`` for more information regarding outputs

    See Also
    --------
    This is no more than a shell around ``sort_points_radial``
    """
    if sort_fields is None:
        print("A structured array with named fields required")
        return a
    v = a[sort_fields]
    idx = np.argsort(v, order=sort_fields)
    return a[idx]


def sort_points(a, as_structured=True):
    """Radial sort of points about the centre point.  The output is a
    structured array or optionally an ndarray.  In both cases, an array of the
    sort order is returned.

    Parameters
    ----------
    a : array
       2D array
    as_structured : boolean
        - True, returns the sorted points and an array of sort indices
        - False, returns a structured array and and an array of sort indices

    Notes
    -----
    More information is returned as well, including the angle from the
    centre to the point.  This angle is relative to the x-axis following
    mathemetical conventions.

    The output dtype for the structured array contains the new and old point
     ids and the angle and distance from the centre.

    >>> a_s, indices = sort_points(a, True)
    >>> a_s.dtype
    dtype=[('ID', '<i4'), ('Old_ID', '<i4'), ('X', '<f8'), ('Y', '<f8'),
           ('Angle', '<f8'), ('Distance', '<f8')]
    """
    a = np.asarray(a)
    if a.ndim != 2:
        msg = "2D array required"
        return a, msg
    cent = a.mean(axis=0)
    ba = a - cent
    ang_ab = np.arctan2(ba[:, 1], ba[:, 0])
    ang_ab = np.degrees(ang_ab)
    dist = np.sqrt(ba[:, 0]**2 + ba[:, 1]**2)
    sort_order = np.argsort(ang_ab)
    if as_structured:
        N = a.shape[0]
        old_id = np.arange(N)
        dt = [('ID', '<i4'), ('Old_ID', '<i4'),
              ('X', '<f8'), ('Y', '<f8'),
              ('Angle', '<f8'), ('Distance', '<f8')]
        args = (sort_order, old_id, a[:, 0], a[:, 1], ang_ab, dist)
        a_0 = np.stack(args, axis=1)
        a_final = a_0[sort_order]
        arr = np.zeros(N, dtype=dt)
        names = arr.dtype.names
        for i, n in enumerate(names):
            arr[n] = a_final[:, i]
    else:
        arr = a[sort_order]
    return arr, sort_order


def view_sort(a):
    """Sort ndarrays by row.  This assumes the array represent coordinates
    and other baggage, in the order that they appear in the row.  It is best
    used for sorting x,y or x,y,z coordinates, using argsort.

    Returns
    -------
    This is a very smart option since if fakes a structured/record array format
    enabling argsort to be used with its `order` option.

    The sorted array and the indices of their original positions in the
    input array.

    Examples
    --------
    >>>       a           view_sort(a)
    ... array([[5, 9],   array([[2, 2],
               [9, 5],          [5, 9],
               [8, 5],          [7, 5],
               [7, 5],          [8, 5],
    ...        [2, 2]])         [9, 5]])
    indices returned:  array([4, 0, 3, 2, 1], dtype=int64))

    See also
    --------
    sort_by_order, xy_sort
    """
    a_view = a.view(a.dtype.descr * a.shape[1])
    idx = np.argsort(a_view, axis=0, order=(a_view.dtype.names)).ravel()
    a = np.ascontiguousarray(a[idx])
    return a, idx


def xy_sort(a):
    """Formally called `view_sort`.  See the documentation there.
    """
    return view_sort(a)


# ---- (2) weave, intermingle ndarrays --------------------------------------
def weave(a, b, match=False, by_row=True, by_cell=False):
    """interweave arrays of the same dimension

    Parameters
    ----------
    a, b : arrays
        Equally shaped arrays, truncation will occur for the longer array
    match : boolean
        - False, slices the array with the greater 1st dimension to match the /
          second
        - True, converts the dtype to `float` and pads the shorter array with
          np.nan
    by_row : boolean
        - For 1D arrays, the weave is by element in the array
        - For N-d arrays, the weave is by dimension, mixing or flattening

    **cool!!!** np.asarray([\*sum(zip(a, b), ())])

    Returns
    -------
    An array of the same dimension is returned

    Examples
    --------
    >>> a = np.arange(8)  # array([0, 1, 2, 3, 4, 5, 6, 7])
    >>> b = a[::-1]       # array([7, 6, 5, 4, 3, 2, 1, 0])
    >>> weave(a, b)  # array([0, 7, 1, 6, 2, 5, 3, 4, 4, 3, 5, 2, 6, 1, 7, 0])
    ... 2D
    >>> # reshape arrays
    >>> a = a.reshape(2, 4);  b = b.reshape(2,4)
    >>> weave(a, b, by_row=True)
    array([[0, 1, 2, 3, 7, 6, 5, 4],
           [4, 5, 6, 7, 3, 2, 1, 0]])
    ...
    >>> weave(a,b, False)
    array([[0, 1, 2, 3],
           [4, 5, 6, 7],
           [7, 6, 5, 4],
           [3, 2, 1, 0]])
    ... 3D
    >>> a = a.reshape(2,2,2);  b = b.reshape(2,2,2)
    >>> a and b         weave(a, b, False)   weave(a, b, True)
    array([[[0, 1],     array([[[0, 1],      array([[[0, 1, 7, 6],
            [2, 3]],            [2, 3],              [2, 3, 5, 4]],
    ...                         [7, 6],      ...
           [[4, 5],             [5, 4]],            [[4, 5, 3, 2],
            [6, 7]]])       ...                      [6, 7, 1, 0]]])
    ...                        [[4, 5],
    array([[[7, 6],             [6, 7],
            [5, 4]],            [3, 2],
    ...                         [1, 0]]])
           [[3, 2],
            [1, 0]]])     Notice the positions of first blocks (0,1,2,3)
    ...  get it now       and (7,6,5,4) when weaving by row or column.

    References
    ----------
    `<https://stackoverflow.com/questions/5347065/interweaving-two-
    numpy-arrays>`_.

    See also
    --------
    - np.stack, np.concatenate
    - a = np.arange(2*5).reshape(5,2)
    - b = np.arange(2*3).reshape(3,2)
    """
    err = "Arrays require the same dimension"
    if match:  # match, pads the shorter array to match the longer array
        arrs = [a, b]
        shps = [a.shape, b.shape]
        s0 = max(shps)
        s1 = min(shps)
        i_min = shps.index(s1)  # i_max = shps.index(s0)
        z = np.full(s0, np.nan).squeeze()
        z[:s1[0], ...] = arrs[i_min]
        if i_min == 1:
            arrs = [a, z]
        else:
            arrs = [z, b]
    else:
        z = np.asarray([*sum(zip(a, b), ())])
        shp = z.shape
        if by_row and z.ndim == 2:
            return z.reshape(shp[-1], np.prod(shp[:-1]))
        return z
    #
    a, b = arrs
    if (a.ndim != b.ndim) or (a.shape != b.shape):
        return err
    dt = np.result_type(a.dtype, b.dtype)
    in_shp = a.shape
    s0 = a.shape[0]
    ndim = a.ndim
    if by_cell:
        s0 = np.product(a.shape)
        a = a.ravel()
        b = b.ravel()
        ndim = 1
    if ndim == 1:
        out = np.zeros((s0*2,), dtype=dt)
        out[0::2] = a
        out[1::2] = b
        if by_cell:
            if by_row:   # keeps first & last
                out = out.reshape(in_shp[0], in_shp[-1], -1)
            else:        # keeps the last 2 dim
                out = out.reshape((-1,) + in_shp[-2:])
        return out
    if by_row:
        return np.concatenate((a, b), axis=-1)
    return np.concatenate((a, b), axis=-2)


def _test_():
    """Test samples
    """
    a = np.array([[8., 1., 6.], [7., 6., 1.], [5., 3., 1.], [3., 0., 3.],
                  [7., 0., 6.], [2., 2., 8.], [3., 0., 4.], [2., 1., 5.],
                  [3., 3., 4.], [7., 2., 8.]])
    z = np.zeros(len(a), dtype="f8, f8, f8")
    names = z.dtype.names
    for i, name in enumerate(names):
        z[name] = a[:, i]
    sort_fields = ['f2', 'f1', 'f0']
    idx = np.argsort(z, order=sort_fields)
    a_s = a[idx]
    z_s = z[idx]
    return (a, z, a_s, z_s, idx)

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    print("Script path {}".format(script))
