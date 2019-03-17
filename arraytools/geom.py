# -*- coding: UTF-8 -*-
"""
====
geom
====

Script :   geom.py

Author :   Dan_Patterson@carleton.ca

Modified : 2019-02-19

Purpose :  tools for working with numpy arrays and geometry

Notes
-----
Do not rely on the OBJECTID field for anything

`<http://support.esri.com/en/technical-article/000010834>`_.

When working with large coordinates, you should check to see whether a
translation about the origin (array - centre) produces different results.
This has been noted when calculating area using projected coordinates for the
Ontario.npy file.  The difference isn't huge, but subtracting the centre or
minimum from the coordinates produces area values which are equal but differ
slightly from those using the unaltered coordinates.

Included in this module::

    'EPSILON', '_arrs_', '_convert', '_new_view_', 'adjacency_edge', 'affine_',
    'angles_poly', 'as_strided', 'cartesian', 'cartesian_dist', 'close_arr',
    'cross', 'dedent', 'densify', 'densify_by_distance', 'densify_by_factor',
    'e_2d', 'e_area', 'e_leng', 'intersect_pnt', 'intersects', 'knn',
    'nn_kdtree', 'np', 'p_o_p', 'pnt_', 'pnt_in_list', 'pnt_on_poly',
    'pnt_on_seg', 'pnts_on_line', 'point_in_polygon', 'poly2segments',
    'remove_self', 'rotate', 'simplify', 'stride', 'trans_rot'

References
----------
See ein_geom.py for full details and examples

`<https://www.redblobgames.com/grids/hexagons/>`_.

`<https://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon>`_.

`<https://iliauk.com/2016/03/02/centroids-and-centres-numpy-r/>`_.

**includes KDTree as well**

`<https://stackoverflow.com/questions/50751135/iterating-operation-with-two-
arrays-using-numpy>`_.

`<https://stackoverflow.com/questions/21483999/using-atan2-to-find-angle-
between-two-vectors>`_.

**point in/on segment**

`<https://stackoverflow.com/questions/328107/how-can-you-determine-a-point
-is-between-two-other-points-on-a-line-segment>`_.

`<https://stackoverflow.com/questions/54442057/calculate-the-euclidian-
distance-between-an-array-of-points-to-a-line-segment-in>`_.

**benchmarking KDTree**

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
from _basic import cartesian
from geom_common import (_new_view_)
from geom_properties import (angles_poly, e_area, e_leng)

EPSILON = sys.float_info.epsilon  # note! for checking

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.2f}'.format}
np.set_printoptions(edgeitems=5, linewidth=160, precision=2, suppress=True,
                    nanstr='nan', infstr='inf',
                    threshold=200, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

# ---- order by appearance ----
#
__all__ = ['close_arr', 'stride',
           'poly2segments', 
           'intersect_pnt', 'intersects',     # intersection
           'cartesian_dist',
           'densify_by_distance',
           'densify_by_factor', '_convert',   # densify simplify
           'densify', 'simplify',
           'rotate', 'trans_rot',             # translation, rotation
           'pnt_in_list',                     # spatial queries and analysis
           'pnt_on_seg', 'pnts_on_line',
           'pnt_on_poly',
           'point_in_polygon',
           'knn', 'nn_kdtree', 'cross',
           'remove_self',
           'adjacency_edge'
           ]


# ---- array functions -------------------------------------------------------
#
def close_arr(a):
    """Close an array representing a sequence of points, so that the
    first and last point are identical.  These arrays are used to construct
    polygons and closed-loop polylines.
    """
    a = np.atleast_2d(a)
    ax = 0 if a.ndim <= 2 else a.ndim-2
    return np.concatenate((a, a[..., :1, :]), axis=ax)


def stride(a, win=(3, 3), stepby=(1, 1)):
    """Provide a 2D sliding/moving view of an array.
    There is no edge correction for outputs. Use the `_pad_` function first.

    Notes
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


def poly2segments(a):
    """Segment poly* structures into o-d pairs from start to finish

    Parameters
    ----------
    a : array
        A 2D array of x,y coordinates representing polyline or polygons.
    fr_to : array
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


# ---- point functions ------------------------------------------------------
#
def intersect_pnt(a, b=None):
    """Returns the point of intersection of the vectors passing through two
    point pairs (p0, p1) and (p2, p3).  This is not segment-segment
    intersection.  An extrapolation will be returned if the segments do not
    cross and the point of intersection will be returned where they **would**
    intersect.

    Parameters
    ----------
    a : array-like
      - 1 segment  [p0, p1]
      - 2 segments [p0, p1], [p2, p3] or
      - 1 array-like np.array([p0, p1, p2, p3])
    b : None or array-like
        1 segment [p2, p3]  if ``a`` is [p0, p1], or ``None``

    Notes
    -----
    Homogenous coordinates are constructed by adding a z/m column equal to 1.
    >>> s = np.array([[ 0,  0], [10, 10], [ 0,  5], [ 5,  0]])
    s:  array([[ 0,  0],    h: array([[  0.,   0.,   1.],
               [10, 10],              [ 10.,  10.,   1.],
               [ 0,  5],              [  0.,   5.,   1.],
               [ 5,  0]])             [  5.,   0.,   1.]])

    References
    ----------
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

    Parameters
    ----------
    - intersects(line0, line1)
    - intersects(p0, p1, p2, p3)

      - p0, p1 -> line 0
      - p2, p3 -> line 1

    Returns
    -------
    boolean, if the segments do intersect

    >>> a = np.array([[0, 0], [10, 10]])
    >>> b = np.array([[0, 10], [10, 0]])
    >>> intersects(*args)  # True

    Examples
    --------
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

    References
    ----------
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


def cartesian_dist(a, b):
    """Form the cartesian product of two 2D arrays.

    Parameters
    ----------
    a, b : arrays
        2D array of x,y values

    Notes
    -----
    The `cartesian` function defined within.  The arrays are passed to it to
    format the data structure to from-to x,y coordinate pairs.

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

# ---- densify functions -----------------------------------------------------
#
def densify_by_factor(a, fact=2):
    """Densify a 2D array using np.interp.

    Parameters
    ----------
    a : array
        Input polyline or polygon array coordinates
    fact : number
        The factor to density the line segments by

    See Also
    --------
    densify_by_distance. This option used an absolute distance separation
    along the segments making up the line feature.

    Notes
    -----
    This is the original construction of c rather than the zero's approach
    outlined in the code which constructs and adds to a zeros array

    >>> c0 = c0.reshape(n, -1)
    >>> c1 = c1.reshape(n, -1)
    >>> c = np.concatenate((c0, c1), 1)
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


def densify_by_distance_old(a, spacing=1):
    """Densify a 2D array by adding points with a specified distance between
    them.  Only appropriate for data representing planar coordinates.

    Parameters
    ----------
    a : array
        A sequence of `points`, x,y pairs, representing the bounds of a polygon
        or polyline object
    spacing : number
        Spacing between the points to be added to the line.

    See Also
    --------
    densify_by_factor, insert_pnts and pnts_on_line

    References
    ----------
    `<https://stackoverflow.com/questions/54665326/adding-points-per-pixel-
    along-some-axis-for-2d-polygon>`_.

    `<https://stackoverflow.com/questions/51512197/python-equidistant-points
    -along-a-line-joining-set-of-points/51514725>`_.

    >>> a = np.array([[0,0], [3, 3], [3, 0], [0,0]])  # 3, 3 triangle
    >>> a = np.array([[0,0], [4, 3], [4, 0], [0,0]])  # 3-4-5 rule
    >>> a = np.array([[0,0], [3, 4], [3, 0], [0,0]])  # 3-4-5 rule
    """
    # ----
    a = _new_view_(a)
    a = np.squeeze(a)
    pnts = []
    for i, e in enumerate(a[1:]):
        s = a[i]                      # 1st point, s
        dxdy = (e - s)                # end - start
        d = np.sqrt(np.sum(dxdy**2))  # end - start distance   
        N = (d/spacing)
        if N < 1:
            pnts.append([s])
        else:
            delta = dxdy/N
            num = np.arange(N)
            delta = np.where(np.isfinite(delta), delta, 0 )
            nn = np.array([num, num]).T
            pnts.append(nn*delta + s)
    pnts.append(a[-1])
    return np.vstack(pnts)


def densify_by_distance(a, spacing):
    """wrapper for `pnts_on_line`

    Example
    -------
    >>> a = np.array([[0., 0.], [3., 4.], [3., 0.], [0., 0.]])  # 3x4x5 rule
    >>> a.T
    array([[0., 3., 3., 0.],
           [0., 4., 0., 0.]])
    >>> pnts_on_line(a, spacing=2).T  # take the transpose to facilitate view
    ... array([[0. , 1.2, 2.4, 3. , 3. , 3. , 1. , 0. ],
    ...        [0. , 1.6, 3.2, 4. , 2. , 0. , 0. , 0. ]])
    ... array([[0.,  . . . .   3., . .   3., . . . 0. ],    
    ...        [0.,  . . . .   4., . .   0., . . . 0. ]])

    >>> letter ``C`` and skinny ``C``
    >>> a = np.array([[ 0, 0], [ 0, 100], [100, 100], [100,  80],
                      [ 20,  80], [ 20, 20], [100, 20], [100, 0], [ 0, 0]])
    >>> b = np.array([[ 0., 0.], [ 0., 10.], [10., 10.], [10.,  8.],
                      [ 2., 8.], [ 2., 2.], [10., 2.], [10., 0.], [ 0., 0.]])
    Notes
    -----
    The return value could be np.vstack((\*pnts, a[-1])) using the last point
    directly, but np.concatenate with a reshaped a[-1] is somewhat faster.
    All entries to the stacking must be ndim=2.

    References
    ----------
    `<https://stackoverflow.com/questions/54665326/adding-points-per-pixel-
    along-some-axis-for-2d-polygon>`_.

    `<https://stackoverflow.com/questions/51512197/python-equidistant-points
    -along-a-line-joining-set-of-points/51514725>`_.
    """
    return pnts_on_line(a, spacing)


def pnts_on_line(a, spacing=1):
    """Add points, at a fixed spacing, to an array representing a line.
    The function ``densify_by_distance`` is a wrapper to this one.

    **See** ``densify_by_distance`` for documentation

    Parameters
    ----------
    a : array
        A sequence of `points`, x,y pairs, representing the bounds of a polygon
        or polyline object
    spacing : number
        Spacing between the points to be added to the line.
    """
    N = len(a) - 1                                    # segments
    dxdy = a[1:, :] - a[:-1, :]                       # coordinate differences
    leng = np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy)) # segment lengths
    steps = leng/spacing                              # step distance
    deltas = dxdy/(steps.reshape(-1, 1))              # coordinate steps
    pnts = np.empty((N,), dtype='O')                  # construct an `O` array
    for i in range(N):              # cycle through the segments and make
        num = np.arange(steps[i])   # the new points
        pnts[i] = np.array((num, num)).T * deltas[i] + a[i]
    a0 = a[-1].reshape(1,-1)        # add the final point and concatenate
    return np.concatenate((*pnts, a0), axis=0)


def _convert(a, fact=2, check_arcpy=True):
    """Do the shape conversion for the array parts.  Calls densify_by_factor

    # import arcpy  # uncomment the first line below if using _convert
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
            shp = densify_by_factor(p, fact=fact)  # call densify_by_factor
            arc_pnts = [Point(*p) for p in shp]
            sub_out.append(arc_pnts)
            out.extend(sub_out)
        else:
            for pp in p:
                shp = densify_by_factor(pp, fact=fact)
                arc_pnts = [Point(*ps) for ps in shp]
                sub_out.append(arc_pnts)
            out.append(sub_out)
    return out


def densify(polys, fact=2):
    """Convert polygon objects to arrays, densify.

    Parameters
    ----------
    densify_by_factor : function
        the function that is called for each shape part
    _unpack : function
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


def affine_(a, center=True, angle=0.0, clockwise=False):
    """Translate and rotate an array of points about its center.

    Parameters
    ----------
    center : boolean
        True, the center point is used as the origin of the axis of rotation.
        False, no translation is used prior to rotation.
    angle : number
        The rotation angle, in degrees, to rotate the points.
    clockwise : boolean
        True, negates the angle for CW rotation.  False leaves the angle along.

    >>> from arraytools.graphing.arr_scatter import plot_pnts_ as pl
    >>> a = np.array([[0, 0], [0, 6], [8, 6], [8, 0], [4, 3]])
    >>> a0 = affine_(a, center=True, angle=22.5, clockwise=False)
    >>> pl([a, b], params=False)

    einsum notes::

        # ---- default, translate to origin, rotate CCW and translate back
        a0 = np.einsum('ij,jk->ik', a-cent, R)
        # ---- alternate if one wants a CW rotation immediately
        a1 = np.einsum('ij,nj->in', a-cent, R)
    """
    if center:
        cent = a.mean(axis=0)
    else:
        cent = np.array([0., 0.])
    if clockwise:
        angle = -angle
    angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, s), (-s, c)))
    return  np.einsum('ij,jk->ik', a-cent, R) + cent


def trans_rot(a, cent=None, angle=0.0, unique=True):
    """Translate and rotate and array of points about the point cloud origin.

    Parameters
    ----------
    a : array
        2d array of x,y coordinates.
    angle : double
        angle in degrees in the range -180. to 180
    unique : boolean
        If True, then duplicate points are removed.  If False, then this would
        be similar to doing a weighting on the points based on location.

    Returns
    -------
    Points rotated about the origin and translated back.

    >>> a = np.array([[0, 0], [0, 6], [8, 6], [8, 0]])  #, [0, 0]])
    >>> b = trans_rot(a, 2.5)
    >>> b
    array([[ 0.5,  1.5],
           [ 1.5,  1.5],
           [ 0.5, -0.5],
           [ 1.5, -0.5]])

    Notes
    -----
    - if the points represent a polygon, make sure that the duplicate
    - np.einsum('ij,kj->ik', a - cent, R)  =  np.dot(a - cent, R.T).T
    - ik does the rotation in einsum

    >>> R = np.array(((c, s), (-s,  c)))  # clockwise about the origin

    *** use this ***
    `<https://stackoverflow.com/questions/54445195/
    np-tensordot-for-rotation-of-point-clouds>`_.
    """
    if unique:
        a = np.unique(a, axis=0)
    if cent is None:
        cent = a.mean(axis=0)
    angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, s), (-s, c)))
    return  np.einsum('ij,jk->ik', a - cent, R) + cent


# ---- points in or on geometries --------------------------------------------
#
def pnt_in_list(pnt, pnts_list):
    """Check to see if a point is in a list of points

    sum([(x, y) == tuple(i) for i in [p0, p1, p2, p3]]) > 0
    """
    is_in = np.any([np.isclose(pnt, i) for i in pnts_list])
    return is_in


def pnt_on_seg(pnt, seg):
    """Orthogonal projection of a point onto a 2 point line segment.
    Returns the intersection point, if the point is between the segment end
    points, otherwise, it returns the distance to the closest endpoint.

    Parameters
    ----------
    pnt : array-like
        `x,y` coordinate pair as list or ndarray
    seg : array-like
        `from-to points`, of x,y coordinates as an ndarray or equivalent

    Notes
    -----
    >>> seg = np.array([[0, 0], [10, 10]])  # p0, p1
    >>> p = [10, 0]
    >>> pnt_on_seg(seg, p)
    array([5., 5.])

    Generically, with crossproducts and norms

    >>> d = np.linalg.norm(np.cross(p1-p0, p0-p))/np.linalg.norm(p1-p0)
    """
    x0, y0, x1, y1, dx, dy = *pnt, *seg[0], *(seg[1] - seg[0])
    dist_ = dx*dx + dy*dy  # squared length
    u = ((x0 - x1)*dx + (y0 - y1)*dy)/dist_
    u = max(min(u, 1), 0)
    xy = np.array([dx, dy])*u + [x1, y1]
    d = xy - pnt
    return xy, np.hypot(d[0], d[1])


def pnt_on_poly(pnt, poly):
    """Find closest point location on a polygon/polyline.

    Parameters
    ----------
    pnt : 1D ndarray array
        XY pair representing the point coordinates.
    poly : 2D ndarray array
        A sequence of XY pairs in clockwise order is expected.  The first and
        last points may or may not be duplicates, signifying sequence closure.

    Returns
    -------
    A list of [x, y, distance, angle] for the intersection point on the line.
    The angle is relative to north from the origin point to the point on the
    polygon.

    Notes
    -----
    e_dist is represented by _e_2d and pnt_on_seg by its equivalent below.
    
    _line_dir_ is from it's equivalent line_dir included here.

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
    def _line_dir_(orig, dest):
        """mini line direction function"""
        orig = np.atleast_2d(orig)
        dest = np.atleast_2d(dest)
        dxy = dest - orig
        ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
        return ang
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
        dest = [n1[0], n1[1]]
        ang = _line_dir_(pnt, dest)
        ang = np.mod((450.0 - ang), 360.)
        return [n1[0], n1[1], np.asscalar(d1), np.asscalar(ang)]
    else:
        dest = [n2[0], n2[1]]
        ang = _line_dir_(pnt, dest)
        ang = np.mod((450.0 - ang), 360.)
        return [n2[0], n2[1], np.asscalar(d2), np.asscalar(ang)]


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

    Parameters
    ----------
    p : array
        x,y reference point
    pnts : array
        Points array to examine
    k : integer
        The `k` in k-nearest neighbours

    Returns
    -------
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

    Parameters
    ----------
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

    References
    ----------
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
  # capital C
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
        a = [a0, a1, a2, a3, a_1a, a_1b, a_2a, a_2b, a_3a, a_3b]
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
    b = densify_by_distance(a, spacing=100)
    fact = 100
#    b = densify_by_factor(a, fact=fact)
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