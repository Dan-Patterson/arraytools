# -*- coding: UTF-8 -*-
"""
geom
====

Script :   geom.py

Author :   Dan_Patterson@carleton.ca

Modified : 2019-01-01

Purpose :  tools for working with numpy arrays and geometry

Notes:
-----
- Do not rely on the OBJECTID field for anything
  http://support.esri.com/en/technical-article/000010834

- When working with large coordinates, you should check to see whether a
  translation about the origin (array - centre) produces different results.
  This has been noted when calculating area using projected coordinates for the
  Ontario.npy file.  The difference isn't huge, but subtracting the centre or
  minimum from the coordinates produces area values which are equal but differ
  slightly from those using the unaltered coordinates.


**Functions**
::
    '__all__', '__builtins__', '__cached__', '__doc__', '__file__',
    '__loader__', '__name__', '__package__', '__spec__', '_arrs_', '_convert',
    '_demo', '_densify_2D', '_flat_', '_new_view_', '_reshape_', '_test',
    '_unpack', '_view_', 'adjacency_edge', 'angle_2pnts', 'angle_between',
    'angle_np', 'angle_seq', 'angles_poly', 'areas', 'as_strided', 'azim_np',
    'center_', 'centers', 'centroid_', 'centroids', 'convex',
    'cross', 'dedent', 'densify', 'dist_bearing', 'dx_dy_np', 'e_2d',
    'e_area', 'e_dist', 'e_leng', 'extent_', 'fill_diagonal', 'ft',
    'intersect_pnt', 'knn', 'lengths', 'line_dir', 'max_',
    'min_', 'nn_kdtree', 'np', 'p_o_p', 'pnt_', 'pnt_in_list', 'pnt_on_poly',
    'pnt_on_seg', 'point_in_polygon', 'radial_sort',
    'remove_self', 'rotate', 'seg_lengths', 'segment', 'simplify', 'stride',
    'total_length', 'trans_rot'

References:
----------
See ein_geom.py for full details and examples

`<https://www.redblobgames.com/grids/hexagons/>`_

`<https://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon>`_

`<https://iliauk.com/2016/03/02/centroids-and-centres-numpy-r/>`_

*includes KDTree as well*

`<https://stackoverflow.com/questions/50751135/iterating-operation-with-two-
arrays-using-numpy>`_

`<https://stackoverflow.com/questions/21483999/using-atan2-to-find-angle-
between-two-vectors>`_

point in/on segment

`<https://stackoverflow.com/questions/328107/how-can-you-determine-a-point
-is-between-two-other-points-on-a-line-segment>`_.

benchmarking KDTree

`<https://www.ibm.com/developerworks/community/blogs/jfp/entry/
Python_Is_Not_C_Take_Two?lang=en>`_.

`<https://iliauk.wordpress.com/2016/02/16/millions-of-distances-high-
performance-python/>`_.

**cKDTree examples**

query_ball_point::

    x, y = np.mgrid[0:3, 0:3]
    pnts = np.c_[x.ravel(), y.ravel()]
    t = cKDTree(pnts)
    idx = t.query_ball_point([1, 0], 1)  # find points within x,y = (1,0)
    pnts[idx]
    array([[0, 0],
    ...    [1, 0],
    ...    [1, 1],
    ...    [2, 0]])

------
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

# ---- imports, formats, constants ----
import sys
from textwrap import dedent
import numpy as np
from numpy.lib.stride_tricks import as_strided
from arraytools._basic import cartesian

EPSILON = sys.float_info.epsilon  # note! for checking

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.2f}'.format}
np.set_printoptions(edgeitems=5, linewidth=120, precision=2, suppress=True,
                    nanstr='nan', infstr='inf',
                    threshold=200, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

# ---- order by appearance ----
#
__all__ = ['_flat_', '_unpack',               # general
           'poly2segments', 'stride',
           '_new_view_', '_view_',
           '_reshape_',
           'min_', 'max_',
           'extent_',                         # extent, centrality
           'center_', 'centroid_',
           'centers', 'centroids',
           'intersect_pnt', 'intersects',     # intersection
           'e_area', 'e_dist', 'e_leng',      # areas, distances, lengths
           'e_2d', 'cartesian_dist',
           'areas', 'lengths',
           'total_length', 'seg_lengths',
           'radial_sort',                     # sorting
           'dist_bearing_sort',
           'dx_dy_np', 'angle_between',       # angles, direction
           'angle_np', 'azim_np',
           'angle_2pnts', 'angle_seq',
           'angles_poly',
           'line_dir',
           '_densify_2D', '_convert',         # densify simplify
           'densify', 'simplify',
           'rotate', 'trans_rot',             # translation, rotation
           'pnt_in_list',                     # spatial queries and analysis
           'pnt_on_seg', 'pnt_on_poly',
           'point_in_polygon',
           'knn', 'nn_kdtree', 'cross', 'remove_self',
           'adjacency_edge'
           ]


# ---- array functions -------------------------------------------------------
#
def _flat_(a_list, flat_list=None):
    """Change the isinstance as appropriate.  Flatten an object using recursion

    see: itertools.chain() for an alternate method of flattening.
    """
    if flat_list is None:
        flat_list = []
    for item in a_list:
        if isinstance(item, (list, tuple, np.ndarray, np.void)):
            _flat_(item, flat_list)
        else:
            flat_list.append(item)
    return flat_list


def _unpack(iterable, param='__iter__'):
    """Unpack an iterable based on the param(eter) condition using recursion.
    From `unpack` in `_common.py`
    """
    xy = []
    for x in iterable:
        if hasattr(x, param):
            xy.extend(_unpack(x))
        else:
            xy.append(x)
    return xy


def poly2segments(a):
    """Segment poly* structures into o-d pairs from start to finish

    `a` : array
        A 2D array of x,y coordinates representing polyline or polygons.
    `fr_to` : array
        Returns a 3D array of point pairs.
    """
    a = _new_view_(a)
    if a.shape[0] == 1:     # squeeze (1, n, m), (n, 1, m) (n, m, 1) arrays
        a = a.squeeze()
    s0, s1 = a.shape
    fr_to = np.zeros((s0-1, s1, 2), dtype=a.dtype)
    fr_to[..., 0] = a[:-1]
    fr_to[..., 1] = a[1:]
    return fr_to


def stride(a, win=(3, 3), stepby=(1, 1)):
    """Provide a 2D sliding/moving view of an array.
    There is no edge correction for outputs. Use the `_pad_` function first.

    Note:
    -----
        Origin arraytools.tools  stride, see it for more information
    """
    err = """Array shape, window and/or step size error.
    Use win=(3,) with stepby=(1,) for 1D array
    or win=(3,3) with stepby=(1,1) for 2D array
    or win=(1,3,3) with stepby=(1,1,1) for 3D
    ----    a.ndim != len(win) != len(stepby) ----
    """
    assert (a.ndim == len(win)) and (len(win) == len(stepby)), err
    shape = np.array(a.shape)  # array shape (r, c) or (d, r, c)
    win_shp = np.array(win)    # window      (3, 3) or (1, 3, 3)
    ss = np.array(stepby)      # step by     (1, 1) or (1, 1, 1)
    newshape = tuple(((shape - win_shp) // ss) + 1) + tuple(win_shp)
    newstrides = tuple(np.array(a.strides) * ss) + a.strides
    a_s = as_strided(a, shape=newshape, strides=newstrides, subok=True).squeeze()
    return a_s


# ---- _view and _reshape_ are helper functions -----------------------------
#
def _new_view_(a):
    """View a structured array x,y coordinates as an ndarray to facilitate
    some array calculations.

    NOTE:  see _view_ for the same functionality
    """
    a = np.asanyarray(a)
    if len(a.dtype) > 1:
        shp = a.shape[0]
        a = a.view(dtype='float64')
        a = a.reshape(shp, 2)
    return a


def _view_(a):
    """Return a view of the array using the dtype and length

    Notes:
    ------
    The is a quick function.  The expectation is that they are coordinate
    values in the form  dtype([('X', '<f8'), ('Y', '<f8')])
    """
    return a.view((a.dtype[0], len(a.dtype.names)))


def _reshape_(a):
    """Reshape arrays, structured or recarrays of coordinates to a 2D ndarray.

    Notes
    -----

    1. The length of the dtype is checked. Only object ('O') and arrays with
       a uniform dtype return 0.  Structured/recarrays will yield 1 or more.

    2. dtypes are stripped and the array reshaped

    >>> a = np.array([(341000., 5021000.), (341000., 5022000.),
                      (342000., 5022000.), (341000., 5021000.)],
                     dtype=[('X', '<f8'), ('Y', '<f8')])
        becomes...
        a = np.array([[  341000.,  5021000.], [  341000.,  5022000.],
                      [  342000.,  5022000.], [  341000.,  5021000.]])
        a.dtype = dtype('float64')

    3. 3D arrays are collapsed to 2D

    >>> a.shape = (2, 5, 2) => np.product(a.shape[:-1], 2) => (10, 2)

    4. Object arrays are processed object by object but assumed to be of a
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


# ---- extent, mins and maxs ------------------------------------------------
# Note:
#     The functions here use _reshape_ to ensure compatability with structured
#  or recarrays.  ndarrays pass through _reshape_ untouched.
#  _view_ or _new_view_ could also be used but are only suited for x,y
#  structured/recarrays
#
def min_(a):
    """Array minimums
    """
    a = _reshape_(a)
    if (a.dtype.kind == 'O') or (len(a.shape) > 2):
        mins = np.asanyarray([i.min(axis=0) for i in a])
    else:
        mins = a.min(axis=0)
    return mins


def max_(a):
    """Array maximums
    """
    a = _reshape_(a)
    if (a.dtype.kind == 'O') or (len(a.shape) > 2):
        maxs = np.asanyarray([i.max(axis=0) for i in a])
    else:
        maxs = a.max(axis=0)
    return maxs


def extent_(a):
    """Array extent values
    """
    a = _reshape_(a)
    if isinstance(a, (list, tuple)):
        a = np.asanyarray(a)
    if (a.dtype.kind == 'O') or (len(a.shape) > 2):
        mins = min_(a)
        maxs = max_(a)
        ret = np.hstack((mins, maxs))
    else:
        L, B = min_(a)
        R, T = max_(a)
        ret = np.asarray([L, B, R, T])
    return ret

# ---- centers --------------------------------------------------------------
def center_(a, remove_dup=True):
    """Return the center of an array. If the array represents a polygon, then
    a check is made for the duplicate first and last point to remove one.
    """
    if a.dtype.kind in ('V', 'O'):
        a = _new_view_(a)
    if remove_dup:
        if np.all(a[0] == a[-1]):
            a = a[:-1]
    return a.mean(axis=0)


def centroid_(a, a_6=None):
    """Return the centroid of a closed polygon.

    `a` : array
        A 2D or more of point coordinates.  You need to keep the duplicate
        first and last point.
    `a_6` : number
        If area has been precalculated, you can use its value.
    `e_area` : function (required)
        Contained in this module.
    """
    if a.dtype.kind in ('V', 'O'):
        a = _new_view_(a)
    x, y = a.T
    t = ((x[:-1] * y[1:]) - (y[:-1] * x[1:]))
    if a_6 is None:
        a_6 = e_area(a) * 6.0  # area * 6.0
    x_c = np.sum((x[:-1] + x[1:]) * t) / a_6
    y_c = np.sum((y[:-1] + y[1:]) * t) / a_6
    return np.asarray([-x_c, -y_c])


def centers(a, remove_dup=True):
    """batch centres (ie _center)
    """
    a = np.asarray(a)
    if a.dtype == 'O':
        tmp = [_reshape_(i) for i in a]
        return np.asarray([center_(i, remove_dup) for i in tmp])
    if len(a.dtype) >= 1:
        a = _reshape_(a)
    if a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    c = np.asarray([center_(i, remove_dup) for i in a]).squeeze()
    return c


def centroids(a):
    """batch centroids (ie _centroid)
    """
    a = np.asarray(a)
    if a.dtype == 'O':
        tmp = [_reshape_(i) for i in a]
        return np.asarray([centroid_(i) for i in tmp])
    if len(a.dtype) >= 1:
        a = _reshape_(a)
    if a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    c = np.asarray([centroid_(i) for i in a]).squeeze()
    return c


# ---- point functions ------------------------------------------------------
#
def intersect_pnt(a, b=None):
    """Returns the point of intersection of the vectors passing through two
    point pairs (p0, p1) and (p2, p3).  This is not segment-segment
    intersection.  An extrapolation will be returned if the segments do not
    cross and the point of intersection will be returned where they **would**
    intersect.

    Parameters:
    -----------
    a : array-like
        1 segment  [p0, p1]
        2 segments [p0, p1], [p2, p3] or
        1 array-like np.array([p0, p1, p2, p3])
    b : None or array-like
        1 segment [p2, p3]  if `a` is [p0, p1], or ``None``

    Notes:
    ------
    >>> s = np.array([[ 0,  0], [10, 10], [ 0,  5], [ 5,  0]])
     s: array([[ 0,  0],    h: array([[  0.,   0.,   1.],
               [10, 10],              [ 10.,  10.,   1.],
               [ 0,  5],              [  0.,   5.,   1.],
               [ 5,  0]])             [  5.,   0.,   1.]])

    Reference:
    ---------
    `<https://stackoverflow.com/questions/3252194/numpy-and-line-
    intersections>`_.
    """
    if (len(a) == 4) and (b is None):
        s = a
    elif (len(a) == 2) and (len(b) == 2):
        s = np.vstack((a, b))
    else:
        raise AttributeError("Use a 4 point array or 2, 2-pnt lines")
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])            # get first line
    l2 = np.cross(h[2], h[3])            # get second line
    x, y, z = np.cross(l1, l2)           # point of intersection
    if z == 0:                           # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)


def intersects(*args):
    """Line intersection check.  Two lines or 4 points that form the lines.

    Requires:
    --------
    - intersects(line0, line1)
    - intersects(p0, p1, p2, p3)

      - p0, p1 -> line 0
      - p2, p3 -> line 1

    Returns:
    --------
    boolean, if the segments do intersect

    >>> a = np.array([[0, 0], [10, 10]])
    >>> b = np.array([[0, 10], [10, 0]])
    >>> intersects(*args)  # True

    Example:
    -------
    ::

        c = np.array([[0, 0], [0, 90], [90, 90], [60, 60], [20, 20], [0, 0]])
        segs = [np.array([c[i-1], c[i]]) for i in range(1, len(c))]
        ln = np.array([[50, -10], [50, 100]])
        print("line {}".format(ln.ravel()))
        for i, j in enumerate(segs):
            r = intersects(ln, j)
            print("{}..{}".format(j.ravel(), r))
        ...
        line [ 50 -10  50 100]
        [ 0  0  0 90]..(False, 'collinear')
        [ 0 90 90 90]..(True, (50.0, 90.0))
        [90 90 60 60]..(False, 'denom check')
        [60 60 20 20]..(True, (50.0, 50))
        [20 20  0  0]..(False, 'cross(p1-p0, p0-p2)')
    References:
    -----------
    `<https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-
    line-segments-intersect#565282>`_.

    """
    if len(args) == 2:    # two lines
        p0, p1, p2, p3 = *args[0], *args[1]
    elif len(args) == 4:  # four points
        p0, p1, p2, p3 = args
    else:
        raise AttributeError("Pass 2, 2-pnt lines or 4 points to the function")
    #
    x0, y0, x1, y1, x2, y2, x3, y3 = *p0, *p1, *p2, *p3  # points to xs and ys
    #
    # ---- First check ----   np.cross(p1-p0, p3-p2 )
    denom = (x1 - x0) * (y3 - y2) - (x3 - x2) * (y1 - y0)
    if denom == 0.0:  # collinear
        return (False, "collinear")
    #
    # ---- Second check ----  np.cross(p1-p0, p0-p2)
    denom_gt0 = denom > 0  # denominator greater than zero
    #
    s_numer = (x1 - x0) * (y0 - y2) - (y1 - y0) * (x0 - x2)
    if (s_numer < 0) == denom_gt0:
        return (False, "cross(p1-p0, p0-p2)")
    #
    # ---- Third check ----  np.cross(p3-p2, p0-p2)
    t_numer = (x3 - x2) * (y0 - y2) - (y3 - y2) * (x0 - x2)
    if (t_numer < 0) == denom_gt0:
        return (False, "cross(p3-p2, p0-p2)")
    #
    # ---- Fourth check ----
    if ((s_numer > denom) == denom_gt0) or ((t_numer > denom) == denom_gt0):
        return (False, "denom check")
    #
    # ---- check to see if the intersection point is one of the input points
    # substitute p0 in the equation  These are the intersection points
    t = t_numer / denom
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)
    # be careful that you are comparing tuples to tuples, lists to lists
    if sum([(x, y) == tuple(i) for i in [p0, p1, p2, p3]]) > 0:
        return (False, "not input point")
    return (True, (x, y))


# ---- distance, length and area --------------------------------------------
# ----
def e_area(a, b=None):
    """Area calculation, using einsum.

    Some may consider this overkill, but consider a huge list of polygons,
    many multipart, many with holes and even multiple version therein.

    Requires:
    --------
    preprocessing :
        use `_view_`, `_new_view_` or `_reshape_` with structured/recarrays
    `a` : array
        Either a 2D+ array of coordinates or arrays of x, y values
    `b` : array, optional
        If a < 2D, then the y values need to be supplied
    Outer rings are ordered clockwise, inner holes are counter-clockwise

    Notes:
    -----
    See ein_geom.py for examples

    """
    a = np.asarray(a)
    if b is None:
        xs = a[..., 0]
        ys = a[..., 1]
    else:
        b = np.asarray(b)
        xs, ys = a, b
    x0 = np.atleast_2d(xs[..., 1:])
    y0 = np.atleast_2d(ys[..., :-1])
    x1 = np.atleast_2d(xs[..., :-1])
    y1 = np.atleast_2d(ys[..., 1:])
    e0 = np.einsum('...ij,...ij->...i', x0, y0)
    e1 = np.einsum('...ij,...ij->...i', x1, y1)
    area = abs(np.sum((e0 - e1)*0.5))
    return area


def e_dist(a, b, metric='euclidean'):
    """Distance calculation for 1D, 2D and 3D points using einsum

    preprocessing :
        use `_view_`, `_new_view_` or `_reshape_` with structured/recarrays

    Parameters:
    -----------
    `a`, `b` : array like
        Inputs, list, tuple, array in 1, 2 or 3D form
    `metric` : string
        euclidean ('e', 'eu'...), sqeuclidean ('s', 'sq'...),

    Notes:
    -----
    mini e_dist for 2d points array and a single point

    >>> def e_2d(a, p):
            diff = a - p[np.newaxis, :]  # a and p are ndarrays
            return np.sqrt(np.einsum('ij,ij->i', diff, diff))

    See also:
    ---------
    cartesian_dist : function
        Produces pairs of x,y coordinates and the distance, without duplicates.
    """
    a = np.asarray(a)
    b = np.atleast_2d(b)
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 1:
        a = a.reshape(1, 1, a.shape[0])
    if a_dim >= 2:
        a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    if b_dim > 2:
        b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
    diff = a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr


def e_leng(a):
    """Length/distance between points in an array using einsum

    Requires:
    --------
    preprocessing :
        use `_view_`, `_new_view_` or `_reshape_` with structured/recarrays
    `a` : array-like
        A list/array coordinate pairs, with ndim = 3 and the minimum
        shape = (1,2,2), eg. (1,4,2) for a single line of 4 pairs

    The minimum input needed is a pair, a sequence of pairs can be used.

    Returns:
    -------
    `length` : float
        The total length/distance formed by the points
    `d_leng` : float
        The distances between points forming the array

        (40.0, [array([[ 10.,  10.,  10.,  10.]])])

    Notes:
    ------
    >>> diff = g[:, :, 0:-1] - g[:, :, 1:]
    >>> # for 4D
    >>> d = np.einsum('ijk..., ijk...->ijk...', diff, diff).flatten()  # or
    >>> d  = np.einsum('ijkl, ijkl->ijk', diff, diff).flatten()
    >>> d = np.sum(np.sqrt(d)
    """
    #
#    d_leng = 0.0
    # ----
    def _cal(diff):
        """ perform the calculation, see above
        """
        d_leng = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff)).squeeze()
        length = np.sum(d_leng.flatten())
        return length, d_leng
    # ----
    diffs = []
    a = np.atleast_2d(a)
    if a.shape[0] == 1:
        return 0.0
    if a.ndim == 2:
        a = np.reshape(a, (1,) + a.shape)
    if a.ndim == 3:
        diff = a[:, 0:-1] - a[:, 1:]
        length, d_leng = _cal(diff)
        diffs.append(d_leng)
    if a.ndim == 4:
        length = 0.0
        for i in range(a.shape[0]):
            diff = a[i][:, 0:-1] - a[i][:, 1:]
            leng, d_leng = _cal(diff)
            diffs.append(d_leng)
            length += leng
    return length, diffs[0]



def cartesian_dist(a, b):
    """Form the cartesian product of two 2D arrays.

    Parameters:
    -----------
    `a`, `b` : arrays
        2D array of x,y values

    Requires:
    ---------
    The `cartesian` function defined within.  The arrays are passed to it to
    format the data structure to from-to x,y coordinate pairs.

    Notes:
    ------
    The cartesian_dist function is about 2X slower than e_dist from
    arraytools.geom.  It depends on whether you need the x,y coordinate pairs
    and the distance or just one of the two forms of the distance matrix.

    >>> a, b
    array([[0., 0.],   array([[3., 3.],
           [1., 1.],          [4., 4.]])
           [2., 2.]])
    cartesian_dist(a,b)
    array([[0.  , 0.  , 3.  , 3.  , 4.24],
           [0.  , 0.  , 4.  , 4.  , 5.66],
           [1.  , 1.  , 3.  , 3.  , 2.83],
           [1.  , 1.  , 4.  , 4.  , 4.24],
           [2.  , 2.  , 3.  , 3.  , 1.41],
           [2.  , 2.  , 4.  , 4.  , 2.83]])
    >>> d = e_dist(a, b)
    array([[4.24, 5.66],   # a0->b0, a0->b1
           [2.83, 4.24],   # a1->b0, a1->b1
           [1.41, 2.83]])  # a2->b0, a2->b1
    >>> d.ravel()  # array([4.24, 5.66, 2.83, 4.24, 1.41, 2.83])
    """
    c = cartesian((a, b))
    diff = c[:, :2] - c[:, 2:]
    d = np.sqrt(np.einsum('ij,ij->i', diff, diff))
    d = d.reshape(len(d), 1)
    return np.concatenate((c, d), axis=1)


# ---- Batch calculations of e_area and e_leng ------------------------------
#
def areas(a):
    """Calls e_area to calculate areas for many types of nested objects.

    This would include object arrays, list of lists and similar constructs.
    Each part is considered separately.

    Returns:
    -------
        A list with one or more areas.
    """
    a = np.asarray(a)
    if a.dtype == 'O':
        tmp = [_reshape_(i) for i in a]
        return [e_area(i) for i in tmp]
    if len(a.dtype) >= 1:
        a = _reshape_(a)
    if a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    a_s = [e_area(i) for i in a]
    return a_s


def lengths(a, prn=False):
    """Calls `e_leng` to calculate lengths for many types of nested objects.
    This would include object arrays, list of lists and similar constructs.
    Each part is considered separately.

    Returns:
    -------
    A list with one or more lengths. `prn=True` for optional printing.
    """
    def _prn_(a_s):
        """optional result printing"""
        hdr = "{!s:<12}  {}\n".format("Tot. Length", "Seg. Length")
        r = ["{:12.3f}  {!r:}".format(*a_s[i]) for i in range(len(a_s))]
        print(hdr + "\n".join(r))
    #
    a = np.asarray(a)
    if a.dtype == 'O':
        tmp = [_reshape_(i) for i in a]
        a_s = [e_leng(i) for i in tmp]
        if prn:
            _prn_(a_s)
        return a_s
    if len(a.dtype) == 1:
        a = _reshape_(a)
    if len(a.dtype) > 1:
        a = _reshape_(a)
    if isinstance(a, (list, tuple)):
        return [e_leng(i) for i in a]
    if a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    a_s = [e_leng(i) for i in a]
    if prn:
        _prn_(a_s)
    return a_s


def total_length(a):
    """Just return total length from 'length' above
    Returns:
    -------
        List of array(s) containing the total length for each object
    """
    a_s = lengths(a)
    result = [i[0] for i in a_s]
    return result[0]


def seg_lengths(a):
    """Just return segment lengths from 'length above.
    Returns:
    -------
        List of array(s) containing the segment lengths for each object
    """
    a_s = lengths(a)
    result = [i[1] for i in a_s]
    return result[0]


# ---- sorting based on geometry --------------------------------------------
#
def radial_sort(pnts, cent=None, distance=True, as_azimuth=False):
    """Sort about the point cloud center or from a given point

    pnts : points
        An array of points (x,y) as array or list
    cent : array-like
        - A list, tuple, array of the center's x,y coordinates.
        - None, the center's coordinate is calculated from the values with
          duplicates removed

    >>> cent = [0, 0] or np.array([0, 0])

    Returns:
    -------
    - The angles in the range -180, 180 x-axis oriented
    - The pnts are NOT returned sorted, you will have to use the sort_order to
      complete the sorting.

    >>> pnts_sorted = pnts[sort_order]
    """
    pnts = _new_view_(pnts)
    if cent is None:
        cent = center_(pnts, remove_dup=False)
    ba = pnts - cent
    ang_ab = np.arctan2(ba[:, 1], ba[:, 0])
    ang_ab = np.degrees(ang_ab)
    sort_order = np.argsort(ang_ab)
    if as_azimuth:
        ang_ab = np.where(ang_ab > 90, 450.0 - ang_ab, 90.0 - ang_ab)
    if distance:
        dist = e_dist(np.array([0, 0]), ba)
        return ang_ab, sort_order, dist
    return ang_ab, sort_order


# ---- angle related functions ----------------------------------------------
# ---- ndarrays and structured arrays ----
#
def dist_bearing_sort(pnts, cent=None, as_struct=False):
    """Sorts points based on a radial sort of a point set (X, Y) relative to
    the center of the points.  The sort is done using the bearing, then the
    distance of a point to the center.

    *** Note done *** move to arraytools.wars when complete

    xy = a0[0][['X', 'Y']]
    pnts = _view_(xy) or....
    s = dist_bearing_sort(a0[0][['X', 'Y']])
    s.shape # (69688, 2)
    11.4 ms ± 272 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
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


def dx_dy_np(a):
    """Sequential difference in the x/y pairs from a table/array

    `a` : ndarray or structured array.
        It is changed as necessary to a 2D array of x/y values.
    _new_view_ : function
        Does the array conversion to 2D from structured if necessary
    """
    a = _new_view_(a)
    out = np.zeros_like(a)
    diff = a[1:] - a[:-1]
    out[1:] = diff
    return out


def angle_np(a, as_degrees=True):
    """Angle between successive points.

    Requires: `dx_dy_np` and hence, `_new_view_`
    """
    diff = dx_dy_np(a)
    angle = np.arctan2(diff[:, 1], diff[:, 0])
    if as_degrees:
        angle = np.rad2deg(angle)
    return angle


def azim_np(a):
    """Return the azimuth/bearing relative to North

    Requires: `angle_np`, which calls `dx_dy_np` which calls `_new_view_`
    """
    s = angle_np(a, as_degrees=True)
    azim = np.where(s <= 0, 90. - s,
                    np.where(s > 90., 450.0 - s, 90.0 - s))
    return azim


def angle_between(p0, p1, p2):
    """angle between 3 sequential points

    >>> p0, p1, p2 = np.array([[0, 0],[1, 1], [1, 0]])
    angle_between(p0, p1, p2)
    (45.0, -135.0, -90.0)
    """
    d1 = p0 - p1
    d2 = p2 - p1
    ang1 = np.arctan2(*d1[::-1])
    ang2 = np.arctan2(*d2[::-1])
    ang = (ang2 - ang1)  # % (2 * np.pi)
    ang, ang1, ang2 = [np.degrees(i) for i in [ang, ang1, ang2]]
    return ang, ang1, ang2


# ---- ndarrays
def angle_2pnts(p0, p1):
    """Two point angle. p0 represents the `from` point and p1 the `to` point.

    >>> angle = atan2(vector2.y, vector2.x) - atan2(vector1.y, vector1.x)

    Accepted answer from the poly_angles link
    """
    p0, p1 = [np.asarray(i) for i in [p0, p1]]
    ba = p1 - p0
    ang_ab = np.arctan2(*ba[::-1])
    return np.rad2deg(ang_ab % (2 * np.pi))


def angle_seq(a):
    """Sequential angles for a points list

    >>> angle = atan2(vector2.y, vector2.x) - atan2(vector1.y, vector1.x)
    Accepted answer from the poly_angles link
    """
    a = _new_view_(a)
    ba = a[1:] - a[:-1]
    ang_ab = np.arctan2(ba[:, 1], ba[:, 0])
    return np.degrees(ang_ab % (2 * np.pi))


def angles_poly(a=None, inside=True, in_deg=True):
    """Sequential 3 point angles from a poly* shape

    a : array
        an array of points, derived from a polygon/polyline geometry
    inside : boolean
        determine inside angles, outside if False
    in_deg : bolean
        convert to degrees from radians

    Notes:
    ------
    General comments
    ::
        2 points - subtract 2nd and 1st points, effectively making the
        calculation relative to the origin and x axis, aka... slope
        n points - sequential angle between 3 points

    Notes to keep
    ::
        *** keep to convert object to array
        a - a shape from the shape field
        a = p1.getPart()
        b = np.asarray([(i.X, i.Y) if i is not None else ()
                       for j in a for i in j])

    Sample data: the letter C

    >>> a = np.array([[ 0, 0], [ 0, 100], [100, 100], [100,  80],
                      [ 20,  80], [ 20, 20], [100, 20], [100, 0], [ 0, 0]])
    >>> angles_poly(a)  # array([ 90.,  90.,  90., 270., 270.,  90.,  90.])
    """
    a = _new_view_(a)
    if len(a) < 2:
        return None
    if len(a) == 2:
        ba = a[1] - a[0]
        return np.arctan2(*ba[::-1])
    a0 = a[0:-2]
    a1 = a[1:-1]
    a2 = a[2:]
    ba = a1 - a0
    bc = a1 - a2
    cr = np.cross(ba, bc)
    dt = np.einsum('ij,ij->i', ba, bc)
    ang = np.arctan2(cr, dt)
    two_pi = np.pi*2.
    if inside:
        ang = np.where(ang < 0, ang + two_pi, ang)
    else:
        ang = np.where(ang > 0, two_pi - ang, ang)
    if in_deg:
        angles = np.degrees(ang)
    return angles


def line_dir(orig, dest, fromNorth=False):
    """Direction of a line given 2 points

    `orig`, `dest` : arrays
        2D arrays of representing the start and end coordinates of two point
        lines.
    `fromNorth` : boolean
        True or False gives angle relative to North or the x-axis.

    Example:
    --------
    >>> orig = np.array([0, 0])
    >>> xy_s = array([[-1,  0,  1,  1,  1,  0, -1, -1],
                      [ 1,  1,  1,  0, -1, -1, -1,  0]])
    >>> dest - xy_s.T
    >>> dir_ = ["From x-axis", "From N."][fromNorth]
    >>> ang = line_dir(orig, dest, fromNorth=False)
    """
    orig = np.atleast_2d(orig)
    dest = np.atleast_2d(dest)
    dxy = dest - orig
    ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
    if fromNorth:
        ang = np.mod((450.0 - ang), 360.)
    return ang


# ---- densify functions -----------------------------------------------------
#
def _densify_2D(a, fact=2):
    """Densify a 2D array using np.interp.

    `a` : array
        Input polyline or polygon array coordinates
    `fact` : number
        The factor to density the line segments by

    Notes:
    -----
        Original construction of c rather than the zero's approach.
    Example
    ::
          c0 = c0.reshape(n, -1)
          c1 = c1.reshape(n, -1)
          c = np.concatenate((c0, c1), 1)
    """
    # Y = a changed all the y's to a
    a = _new_view_(a)
    a = np.squeeze(a)
    n_fact = len(a) * fact
    b = np.arange(0, n_fact, fact)
    b_new = np.arange(n_fact - 1)     # Where you want to interpolate
    c0 = np.interp(b_new, b, a[:, 0])
    c1 = np.interp(b_new, b, a[:, 1])
    n = c0.shape[0]
    c = np.zeros((n, 2))
    c[:, 0] = c0
    c[:, 1] = c1
    return c


def _convert(a, fact=2, check_arcpy=True):
    """Do the shape conversion for the array parts.  Calls _densify_2D

    Requires:
    ---------
    >>> import arcpy  # uncomment the first line below if using _convert
    """
    if check_arcpy:
        #import arcpy
        from arcpy.arcobjects import Point
    out = []
    parts = len(a)
    for i in range(parts):
        sub_out = []
        p = np.asarray(a[i]).squeeze()
        if p.ndim == 2:
            shp = _densify_2D(p, fact=fact)  # call _densify_2D
            arc_pnts = [Point(*p) for p in shp]
            sub_out.append(arc_pnts)
            out.extend(sub_out)
        else:
            for pp in p:
                shp = _densify_2D(pp, fact=fact)
                arc_pnts = [Point(*ps) for ps in shp]
                sub_out.append(arc_pnts)
            out.append(sub_out)
    return out


def densify(polys, fact=2):
    """Convert polygon objects to arrays, densify.

    Requires:
    --------
    `_densify_2D` : function
        the function that is called for each shape part
    `_unpack` : function
        unpack objects
    """
    # ---- main section ----
    out = []
    for poly in polys:
        p = poly.__geo_interface__['coordinates']
        back = _convert(p, fact)
        out.append(back)
    return out


# ---- simplify functions -----------------------------------------------------
#
def simplify(a, deviation=10):
    """Simplify array
    """
    angles = angles_poly(a, inside=True, in_deg=True)
    idx = (np.abs(angles - 180.) >= deviation)
    sub = a[1: -1]
    p = sub[idx]
    return a, p, angles


# ---- Create geometries -----------------------------------------------------
#
def pnt_(p=np.nan):
    """Create a point object for null points, center points etc
    ::
        pnt_((1., 2.)\n
        pnt_(1) => array([1., 1.])
    """
    p = np.atleast_1d(p)
    if p.dtype.kind not in ('f', 'i'):
        raise ValueError("Numeric points supported, not {}".format(p))
    if np.any(np.isnan(p)):
        p = np.array([np.nan, np.nan])
    elif isinstance(p, (np.ndarray, list, tuple)):
        if len(p) == 2:
            p = np.array([p[0], p[1]])
        else:
            p = np.array([p[0], p[0]])
    return p


def rotate(pnts, angle=0):
    """Rotate points about the origin in degrees, (+ve for clockwise) """
    pnts = _new_view_(pnts)
    angle = np.deg2rad(angle)                 # convert to radians
    s = np.sin(angle)
    c = np.cos(angle)    # rotation terms
    aff_matrix = np.array([[c, s], [-s, c]])  # rotation matrix
    XY_r = np.dot(pnts, aff_matrix)           # numpy magic to rotate pnts
    return XY_r


def trans_rot(a, angle=0.0, unique=True):
    """Translate and rotate and array of points about the point cloud origin.

    Requires:
    ---------
    a : array
        2d array of x,y coordinates.
    angle : double
        angle in degrees in the range -180. to 180
    unique : boolean
        If True, then duplicate points are removed.  If False, then this would
        be similar to doing a weighting on the points based on location.

    Returns:
    --------
    Points rotated about the origin and translated back.

    >>> a = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    >>> b = trans_rot(b, 45)
    >>> b
    array([[ 0.5,  1.5],
           [ 1.5,  1.5],
           [ 0.5, -0.5],
           [ 1.5, -0.5]])

    Notes:
    ------
    - if the points represent a polygon, make sure that the duplicate
    - np.einsum('ij,kj->ik', a - cent, R)  =  np.dot(a - cent, R.T).T
    - ik does the rotation in einsum

    >>> R = np.array(((c, s), (-s,  c)))  # clockwise about the origin
    """
    if unique:
        a = np.unique(a, axis=0)
    cent = a.mean(axis=0)
    angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, s), (-s, c)))
    return  np.einsum('ij,kj->ik', a - cent, R) + cent


# ---- points in or on geometries --------------------------------------------
#
def pnt_in_list(pnt, pnts_list):
    """Check to see if a point is in a list of points

    sum([(x, y) == tuple(i) for i in [p0, p1, p2, p3]]) > 0
    """
    is_in = np.any([np.isclose(pnt, i) for i in pnts_list])
    return is_in


def point_in_polygon(pnt, poly):  # pnt_in_poly(pnt, poly):  #
    """Point is in polygon. ## fix this and use pip from arraytools
    """
    x, y = pnt
    N = len(poly)
    for i in range(N):
        x0, y0, xy = [poly[i][0], poly[i][1], poly[(i + 1) % N]]
        c_min = min([x0, xy[0]])
        c_max = max([x0, xy[0]])
        if c_min < x <= c_max:
            p = y0 - xy[1]
            q = x0 - xy[0]
            y_cal = (x - x0) * p / q + y0
            if y_cal < y:
                return True
    return False


def pnt_on_seg(pnt, seg):
    """Orthogonal projection of a point onto a 2 point line segment
    Returns the intersection point, if the point is between the segment end
     points, otherwise, it returns the distance to the closest endpoint.

    Parameters:
    -----------
    pnt : array-like
        `x,y` coordinate pair as list or ndarray
    seg : array-like
        `from-to points`, of x,y coordinates as an ndarray or equivalent

    Notes:
    ------
    >>> seg = np.array([[0, 0], [10, 10]])  # p0, p1
    >>> p = [10, 0]
    >>> pnt_on_seg(seg, p)
    array([5., 5.])

    d = np.linalg.norm(np.cross(p1-p0, p0-p))/np.linalg.norm(p1-p0)
    """
    x0, y0, x1, y1, dx, dy = *pnt, *seg[0], *(seg[1] - seg[0])
    dist_ = dx*dx + dy*dy  # squared length
    u = ((x0 - x1)*dx + (y0 - y1)*dy)/dist_
    u = max(min(u, 1), 0)
    xy = np.array([dx, dy])*u + [x1, y1]
    return xy, np.sqrt(dist_)


def pnt_on_poly(pnt, poly):
    """Find closest point location on a polygon/polyline.

    Parameters:
    -----------
    pnt : 1D ndarray array
        XY pair representing the point coordinates.
    poly : 2D ndarray array
        A sequence of XY pairs in clockwise order is expected.  The first and
        last points may or may not be duplicates, signifying sequence closeure.

    Returns:
    --------
    A list of [x, y, distance] for the intersection point on the line

    Requires:
    ---------
    e_dist is represented by _e_2d and pnt_on_seg by its equivalent below.

    Notes:
    ------
    This may be as simple as finding the closest point on the edge, but if
    needed, an orthogonal projection onto a polygon/line edge will be done.
    This situation arises when the distance to two sequential points is the
    same
    """
    def _e_2d_(a, p):
        """ array points to point distance... mini e_dist"""
        diff = a - p[np.newaxis, :]
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))
    #
    def _pnt_on_seg_(seg, pnt):
        """mini pnt_on_seg function normally required by pnt_on_poly"""
        x0, y0, x1, y1, dx, dy = *pnt, *seg[0], *(seg[1] - seg[0])
        dist_ = dx*dx + dy*dy  # squared length
        u = ((x0 - x1)*dx + (y0 - y1)*dy)/dist_
        u = max(min(u, 1), 0)  # u must be between 0 and 1
        xy = np.array([dx, dy])*u + [x1, y1]
        return xy
    #
    pnt = np.asarray(pnt)
    poly = np.asarray(poly)
    if np.all(poly[0] == poly[-1]):  # strip off any duplicate
        poly = poly[:-1]
    # ---- determine the distances
    d = _e_2d_(poly, pnt)  # abbreviated edist =>  d = e_dist(poly, pnt)
    key = np.argsort(d)[0]         # dist = d[key]
    if key == 0:
        seg = np.vstack((poly[-1:], poly[:3]))
    elif (key + 1) >= len(poly):
        seg = np.vstack((poly[-2:], poly[:1]))
    else:
        seg = poly[key-1:key+2]    # grab the before and after closest
    n1 = _pnt_on_seg_(seg[:-1], pnt)  # abbreviated pnt_on_seg
    d1 = np.linalg.norm(n1 - pnt)
    n2 = _pnt_on_seg_(seg[1:], pnt)   # abbreviated pnt_on_seg
    d2 = np.linalg.norm(n2 - pnt)
    if d1 <= d2:
        return [n1[0], n1[1], np.asscalar(d1)]
    return [n2[0], n2[1], np.asscalar(d2)]


def p_o_p(pnt, polys):
    """ main runner
    """
    result = []
    for p in polys:
        result.append(pnt_on_poly(p, pnt))
    return result


# ---- nearest neighbors, knn ------------------------------------------------
#
def knn(p, pnts, k=1, return_dist=True):
    """
    Calculates k nearest neighbours for a given point.

    Parameters:
    -----------
    p :array
        x,y reference point
    pnts : array
        Points array to examine
    k : integer
        The `k` in k-nearest neighbours

    Returns:
    --------
    Array of k-nearest points and optionally their distance from the source.
    """
    def _remove_self_(p, pnts):
        """Remove a point which is duplicated or itself from the array
        """
        keep = ~np.all(pnts == p, axis=1)
        return pnts[keep]
    #
    def _e_2d_(p, a):
        """ array points to point distance... mini e_dist
        """
        diff = a - p[np.newaxis, :]
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))
    #
    p = np.asarray(p)
    k = max(1, min(abs(int(k)), len(pnts)))
    pnts = _remove_self_(p, pnts)
    d = _e_2d_(p, pnts)
    idx = np.argsort(d)
    if return_dist:
        return pnts[idx][:k], d[idx][:k]
    return pnts[idx][:k]


def nn_kdtree(a, N=1, sorted_=True, to_tbl=True, as_cKD=True):
    """Produce the N closest neighbours array with their distances using
    scipy.spatial.KDTree as an alternative to einsum.

    Parameters:
    -----------
    a : array
        Assumed to be an array of point objects for which `nearest` is needed.
    N : integer
        Number of neighbors to return.  Note: the point counts as 1, so N=3
        returns the closest 2 points, plus itself.
        For table output, max N is limited to 5 so that the tabular output
        isn't ridiculous.
    sorted_ : boolean
        A nice option to facilitate things.  See `xy_sort`.  Its mini-version
        is included in this function.
    to_tbl : boolean
        Produce a structured array output of coordinate pairs and distances.
    as_cKD : boolean
        Whether to use the `c` compiled or pure python version

    References:
    -----------
    `<https://stackoverflow.com/questions/52366421/how-to-do-n-d-distance-
    and-nearest-neighbor-calculations-on-numpy-arrays/52366706#52366706>`_.

    `<https://stackoverflow.com/questions/6931209/difference-between-scipy-
    spatial-kdtree-and-scipy-spatial-ckdtree/6931317#6931317>`_.
    """
    def _xy_sort_(a):
        """mini xy_sort"""
        a_view = a.view(a.dtype.descr * a.shape[1])
        idx = np.argsort(a_view, axis=0, order=(a_view.dtype.names)).ravel()
        a = np.ascontiguousarray(a[idx])
        return a, idx
    #
    def xy_dist_headers(N):
        """Construct headers for the optional table output"""
        vals = np.repeat(np.arange(N), 2)
        names = ['X_{}', 'Y_{}']*N + ['d_{}']*(N-1)
        vals = (np.repeat(np.arange(N), 2)).tolist() + [i for i in range(1, N)]
        n = [names[i].format(vals[i]) for i in range(len(vals))]
        f = ['<f8']*N*2 + ['<f8']*(N-1)
        return list(zip(n, f))
    #
    from scipy.spatial import cKDTree, KDTree
    #
    if sorted_:
        a, _ = _xy_sort_(a)
    # ---- query the tree for the N nearest neighbors and their distance
    if as_cKD:
        t = cKDTree(a)
    else:
        t = KDTree(a)
    dists, indices = t.query(a, N+1)  # so that point isn't duplicated
    dists = dists[:, 1:]               # and the array is 2D
    frumXY = a[indices[:, 0]]
    indices = indices[:, 1:]
    if to_tbl and (N <= 5):
        dt = xy_dist_headers(N+1)  # --- Format a structured array header
        xys = a[indices]
        new_shp = (xys.shape[0], np.prod(xys.shape[1:]))
        xys = xys.reshape(new_shp)
        #ds = dists[:, 1]  # [d[1:] for d in dists]
        arr = np.concatenate((frumXY, xys, dists), axis=1)
        z = np.zeros((xys.shape[0],), dtype=dt)
        names = z.dtype.names
        for i, j in enumerate(names):
            z[j] = arr[:, i]
        return z
    dists = dists.view(np.float64).reshape(dists.shape[0], -1)
    return dists


# ---- mini stuff -----------------------------------------------------------
#
def cross(o, a, b):
    """Cross-product for vectors o-a and o-b
    """
    xo, yo = o
    xa, ya = a
    xb, yb = b
    return (xa - xo)*(yb - yo) - (ya - yo)*(xb - xo)


def e_2d(p, a):
    """ array points to point distance... mini e_dist
    """
    p = np.asarray(p)
    diff = a - p[np.newaxis, :]
    return np.sqrt(np.einsum('ij,ij->i', diff, diff))


def remove_self(p, pnts):
    """Remove a point which is duplicated or itself from the array
    """
    keep = ~np.all(pnts == p, axis=1)
    return pnts[keep]


def adjacency_edge():
    """keep... adjacency-edge list association
    : the columns are 'from', the rows are 'to' the cell value is the 'weight'.
    """
    adj = np.random.randint(1, 4, size=(5, 5))
    r, c = adj.shape
    mask = ~np.eye(r, c, dtype='bool')
    # dt = [('Source', '<i4'), ('Target', '<i4'), ('Weight', '<f8')]
    m = mask.ravel()
    XX, YY = np.meshgrid(np.arange(c), np.arange(r))
    # mmm = np.stack((m, m, m), axis=1)
    XX = np.ma.masked_array(XX, mask=m)
    YY = np.ma.masked_array(YY, mask=m)
    edge = np.stack((XX.ravel(), YY.ravel(), adj.ravel()), axis=1)
    frmt = """
    Adjacency edge list association...
    {}
    edge...
    (col, row, value)
    {}
    """
    print(dedent(frmt).format(adj, edge))


# ---- Extras ----------------------------------------------------------------
#
def _test(a0=None):
    """testing stuff using a0"""
    import math
    if a0 is None:
        a0 = np.array([[10, 10], [10, 20], [20, 20], [10, 10]])
    x0, y0 = p0 = a0[-2]
    p1 = a0[-1]
    dx, dy = p1 - p0
    dist = math.hypot(dx, dy)
    xc, yc = pc = p0 + (p1 - p0)/2.0
    slope = math.atan2(dy, dx)
    step = 2.
    xn = x0 + math.cos(slope) * step  # dist / fact
    yn = y0 + math.sin(slope) * step  # dist / fact
    # better
    start = 0
    step = 2.0
    stop = 10.0 + step/2
    x2 = np.arange(start, stop + step, step)
    return a0, dist, xc, yc, pc, slope, xn, yn, x2


# ---- data and demos --------------------------------------------------------
#
def _arrs_(prn=True):
    """Sample arrays to test various cases
    """
    cw = np.array([[0, 0], [0, 100], [100, 100], [100, 80], [20, 80],
                   [20, 20], [100, 20], [100, 0], [0, 0]])  # capital C
    a0 = np.array([[10, 10], [10, 20], [20, 20], [10, 10]])
    a1 = np.array([[20., 20.], [20., 30.], [30., 30.], [30., 20.], [20., 20.]])
    a2 = np.array([(20., 20.), (20., 30.), (30., 30.), (30., 20.), (20., 20.)],
                  dtype=[('X', '<f8'), ('Y', '<f8')])
    a3 = np.array([([20.0, 20.0],), ([20.0, 30.0],), ([30.0, 30.0],),
                   ([30.0, 20.0],), ([20.0, 20.0],)],
                  dtype=[('Shape', '<f8', (2,))])
    a_1a = np.asarray([a1, a1])
    a_1b = np.asarray([a1, a1[:-1]])
    a_2a = np.asarray([a2, a2])
    a_2b = np.asarray([a2, a2[:-1]])
    a_3a = np.asarray([a3, a3])
    a_3b = np.asarray([a3, a3[:-1]])
    a0 = a0 - [10, 10]
    a1 = a1 - [10, 10]
    a = [a0, a1, a2, a3, a_1a, a_1b, a_2a, a_2b, a_3a, a_3b]
    sze = [i.size for i in a]
    shp = [len(i.shape) for i in a]
    dtn = [len(i.dtype) for i in a]
    if prn:
        a = [a0, a1, a2, a3, a_1a, a_1b, a_2a, a_2b, a_3a, a_3b, cw]
        n = ['a0', 'a1', 'a2', 'a3',
             'a_1a', 'a_1b',
             'a_2a', 'a_2b',
             'a_3a', 'a_3b']
        args = ['array', 'kind', 'size', 'ndim', 'shape', 'dtype']
        frmt = "{!s:<6} {!s:<5} {!s:<5} {!s:<4} {!s:<10} {!s:<20}"
        print(frmt.format(*args))
        cnt = 0
        for i in a:
            args = [n[cnt], i.dtype.kind, i.size, i.ndim, i.shape, i.dtype]
            print(dedent(frmt).format(*args))
            cnt += 1
    a.extend([sze, shp, dtn])
    return a


def _demo(prn=True):
    """Demo the densify function using Ontario boundary polyline
    :
    : ---- Ontario boundary polyline, shape (49,874, 2) ----
    :  x = "...script location.../Data/Ontario.npy"
#   : a = np.load(x)
    : alternates... but slower
    : def PolyArea(x, y):
    :     return 0.5*np.abs(np.dot(x, np.roll(y,1))-np.dot(y, np.roll(x,1)))

    """
    x = "/".join(script.split("/")[:-1]) + "/Data/Ontario.npy"
    a = np.load(x)
    fact = 2
    b = _densify_2D(a, fact=fact)
    t0, avl = e_leng(a)  # first is total, 2nd is all lengths
    t1 = t0/1000.
    min_l = avl.min()
    avg_l = avl.mean()
    max_l = avl.max()
    ar = e_area(a)
    ar1 = ar/1.0e04
    ar2 = ar/1.0e06
    if prn:
        frmt = """
        Original number of points... {:,}
        Densified by a factor of ... {}
        New point count ............ {:,}
        Ontario perimeter .......... {:,.1f} m  {:,.2f} km
        Segments lengths ... min  {:,.2f} m
                             mean {:,.2f} m
                             max  {:,.2f} m
        Ontario area ............... {:,.2f} ha.  {:,.1f} sq.km.
        """
        args = [a.shape[0], fact, b.shape[0], t0, t1, min_l,
                avg_l, max_l, ar1, ar2]
        print(dedent(frmt).format(*args))
    return a


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    # print the script source name.
    print("Script... {}".format(script))

# sample array... big C and points
#c = np.array([[ 0, 0], [ 0, 1000], [1000, 1000], [1000,  800],
#              [ 200,  800], [ 200, 200], [1000, 200], [1000, 0], [ 0, 0]])
# pnts = np.random.randint(0, 1000, size=(1000,2))

# for Ontario_LCC
# total_length(a)  #: 6804096.2018476073  same as arcmap
# areas(a)  #: [1074121438784.0]  1074121438405.34021

#    from arraytools.fc_tools import fc
#    in_fc = r'C:\Git_Dan\a_Data\arcpytools_demo.gdb\Can_geom_sp_LCC'
#    in_fc = r'C:\Git_Dan\a_Data\arcpytools_demo.gdb\Ontario_LCC'
#    in_fc = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb\Carp_5x5"   # full 25 polygons
#    a = fc._xy(in_fc)
#    a_s = group_pnts(a, key_fld='IDs', shp_flds=['Xs', 'Ys'])
#    a_s = np.asarray(a_s)
#    a_area = areas(a_s)
#    a_tot_leng = total_length(a_s)
#    a_seg_leng = seg_lengths(a_s)
#    a_ng = angles_poly(a1)
#    v = r'C:\Git_Dan\arraytools\Data\sample_100K.npy'  # 20, 1000, 10k, 100K
#    oa = obj_array(in_fc)
#    ta = _two_arrays(in_fc, both=True, split=True)

"""
test for segment segment intersection

a = np.array([[0,0], [3,3]])
b = np.array([[0,3], [3,0]])
p0, p1, p2, p3 = *a, *b
x0, y0, x1, y1, x2, y2, x3, y3 = *p0, *p1, *p2, *p3
denom = (x1 - x0) * (y3 - y2) - (x3 - x2) * (y1 - y0)
denom_gt0 = denom > 0
s_numer = (x1 - x0) * (y0 - y2) - (y1 - y0) * (x0 - x2)
(s_numer < 0) == denom_gt0
t_numer = (x3 - x2) * (y0 - y2) - (y3 - y2) * (x0 - x2)
(t_numer < 0) == denom_gt0
((s_numer > denom) == denom_gt0) or ((t_numer > denom) == denom_gt0)
t = t_numer / denom
x = x0 + t * (x1 - x0)
y = y0 + t * (y1 - y0)
print("intersecton pnt ({},{} t {})".format(x, y, t))
intersecton pnt (1.5,1.5 t 0.5)
"""