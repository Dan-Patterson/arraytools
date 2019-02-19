# -*- coding: utf-8 -*-
"""
===============
geom_properties
===============

Script :   geom_properties.py

Author :   Dan_Patterson@carleton.ca

Modified : 2019-02-18

Purpose :  Properties for geometry objects represented as arrays

Notes

References

Included in this module::

    '_new_view_', '_reshape_', 'angle_2pnts', 'angle_between', 'angle_np',
    'angle_seq', 'angles_poly', 'areas', 'azim_np', 'center_', 'centers',
    'centroid_', 'centroids', 'dx_dy_np', 'e_area', 'e_dist', 'e_leng',
    'extent_', 'ft', 'lengths', 'orig_dest_angle', 'line_dir', 'max_',
    'mean_', 'median_', 'min_', 'np', 'seg_lengths', 'sys', 'total_length'


"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import warnings
import numpy as np
from geom_common import (_new_view_, _reshape_)

warnings.simplefilter('ignore', FutureWarning)

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


__all__ = ['max_', 'median_', 'min_',      # max, mean, median, min
           'extent_',
           'center_', 'centers',           # centers, centroids
           'centroid_', 'centroids',
           'e_area', 'e_dist', 'e_leng',   # areas, distances, lengths
           'areas', 'lengths', 
           'total_length', 'seg_lengths',
           'dx_dy_np', 'angle_np',         # angles, direction
           'azim_np',  'angle_between',
           'angle_2pnts', 'angle_seq',
           'angles_poly',
           'orig_dest_angle', 'line_dir'       
           ]
# ===========================================================================
# Note:
#     The functions here use _reshape_ to ensure compatability with structured
#  or recarrays.  ndarrays pass through _reshape_ untouched.
#  _view_ or _new_view_ could also be used but are only suited for x,y
#  structured/recarrays
#
# ---- stats/descriptive related ---------------------------------------------
def max_(a):
    """Array maximums.  No `finite_check`. See Note above
    """
    a = _reshape_(a)
    if (a.dtype.kind == 'O') or (len(a.shape) > 2):
        maxs = np.asanyarray([i.max(axis=0) for i in a])
    else:
        maxs = a.max(axis=0)
    return maxs


def mean_(a):
    """Array mean. No `finite_check`. See Note above
    """
    a = _reshape_(a)
    if (a.dtype.kind == 'O') or (len(a.shape) > 2):
        means = np.asanyarray([i.mean(axis=0) for i in a])
    else:
        means = a.means(axis=0)
    return means


def median_(a):
    """Array median. No `finite_check`. See Note above
    """
    a = _reshape_(a)
    if (a.dtype.kind == 'O') or (len(a.shape) > 2):
        meds = np.asanyarray([np.median(i, axis=0) for i in a])
    else:
        meds = np.median(a, axis=0)
    return meds


def min_(a):
    """Array minimums. No `finite_check`. See Note above
    """
    a = _reshape_(a)
    if (a.dtype.kind == 'O') or (len(a.shape) > 2):
        mins = np.asanyarray([i.min(axis=0) for i in a])
    else:
        mins = a.min(axis=0)
    return mins


# ---- bounds ---------------------------------------------------------------
def extent_(a):
    """Array extent values. See Note above
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

    See Note above
    """
    if a.dtype.kind in ('V', 'O'):
        a = _new_view_(a)
    if remove_dup:
        if np.all(a[0] == a[-1]):
            a = a[:-1]
    return a.mean(axis=0)


def centroid_(a, a_6=None):
    """Return the centroid of a closed polygon.

    Parameters
    ----------
    a : array
        A 2D or more of point coordinates.  You need to keep the duplicate
        first and last point.
    a_6 : number
        If area has been precalculated, you can use its value.
    e_area : function (required)
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

# ---- distance, length and area --------------------------------------------
# ----
def e_area(a, b=None):
    """Area calculation, using einsum.

    Some may consider this overkill, but consider a huge list of polygons,
    many multipart, many with holes and even multiple version therein.

    Parameters
    ----------
    preprocessing :
        use `_view_`, `_new_view_` or `_reshape_` with structured/recarrays
    a : array
        Either a 2D+ array of coordinates or arrays of x, y values
    b : array, optional
        If a < 2D, then the y values need to be supplied

    Notes
    -----
    Outer rings are ordered clockwise, inner holes are counter-clockwise.
    First and last points are assumed to be the same, if not... fix it.

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

    Parameters
    ----------
    a, b : array like
        Inputs, list, tuple, array in 1, 2 or 3D form
    metric : string
        euclidean ('e', 'eu'...), sqeuclidean ('s', 'sq'...),

    Notes
    -----
    mini e_dist for 2d points array and a single point

    >>> def e_2d(a, p):
            diff = a - p[np.newaxis, :]  # a and p are ndarrays
            return np.sqrt(np.einsum('ij,ij->i', diff, diff))

    See Also
    --------
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


def e_leng(a, close=False):
    """Length/distance between points in an array using einsum

    Parameters
    ----------
    preprocessing :
        use `_view_`, `_new_view_` or `_reshape_` with structured/recarrays
    a : array-like
        A list/array coordinate pairs, with ndim = 3 and the minimum
        shape = (1,2,2), eg. (1,4,2) for a single line of 4 pairs

    The minimum input needed is a pair, a sequence of pairs can be used.

    Returns
    -------
    length : float
        The total length/distance formed by the points
    d_leng : float
        The distances between points forming the array

        (40.0, [array([[ 10.,  10.,  10.,  10.]])])

    Notes
    -----
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
    def _close_ply(a):
        """close an open polyline"""
        if not np.all(a[0] ==  a[-1]):
            a = np.vstack((a, a[0]))
        return a
    # ----
    diffs = []
    a = np.atleast_2d(a)
    if a.shape[0] == 1:
        return 0.0
    if a.ndim == 2:
        if close:
            a = _close_ply(a)
        a = np.reshape(a, (1,) + a.shape)
    if a.ndim == 3:
        if close:
            a = np.array([_close_ply(i) for i in a])
        diff = a[:, 0:-1] - a[:, 1:]
        length, d_leng = _cal(diff)
        diffs.append(d_leng)
    if a.ndim == 4:
        length = 0.0
        if close:
            tmp = []
            for i in a:
                tmp.append(np.array([_close_ply(j) for j in i]))
            a = np.array(tmp)
        for i in range(a.shape[0]):
            diff = a[i][:, 0:-1] - a[i][:, 1:]
            leng, d_leng = _cal(diff)
            diffs.append(d_leng)
            length += leng
    return length, diffs[0]

# ---- Batch calculations of e_area and e_leng ------------------------------
#
def areas(a):
    """Calls e_area to calculate areas for many types of nested objects.

    This would include object arrays, list of lists and similar constructs.
    Each part is considered separately.

    Returns
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
    a_s = np.array([e_area(i) for i in a])
    return a_s


def lengths(a, close=False, prn=False):
    """Calls `e_leng` to calculate lengths for many types of nested objects.
    This would include object arrays, list of lists and similar constructs.
    Each part is considered separately.
    
    Parameters
    ----------
    a : array-like
        An array like sequence of points that are assumed to make up lengths or
        perimeters of geometries
    close : boolean
        True, computes the closed-loop length for polylines and check to ensure
        polygons have duplicate first and last points.
    prn : boolean
        True, prints the output as well as returning the result

    Notes
    -----
    e_leng is required by this function.  Make sure you ensure that you are
    aware of the `close` parameter.

    Returns
    -------
    A list with total length and segment lengths, for multipoint sequences.
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
        a_s = [e_leng(i, close) for i in tmp]
        if prn:
            _prn_(a_s)
        return a_s
    if len(a.dtype) == 1:
        a = _reshape_(a)
    if len(a.dtype) > 1:
        a = _reshape_(a)
    if isinstance(a, (list, tuple)):
        return [e_leng(i, close) for i in a]
    if a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    a_s = [e_leng(i, close) for i in a]
    if prn:
        _prn_(a_s)
    return a_s


def total_length(a):
    """Just return total length from 'length' above

    Returns
    -------
        List of array(s) containing the total length for each object
    """
    a_s = lengths(a)
    result = [i[0] for i in a_s]
    return result[0]


def seg_lengths(a):
    """Just return segment lengths from 'length above.

    Returns
    -------
    List of array(s) containing the segment lengths for each object
    """
    a_s = lengths(a)
    result = [i[1] for i in a_s]
    return result[0]


# ---- angle related functions ----------------------------------------------
# ---- ndarrays and structured arrays ----

def dx_dy_np(a):
    """Sequential difference in the x/y pairs from a table/array

    a : ndarray or structured array.
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

    Requires, `dx_dy_np` and hence, `_new_view_`
    """
    diff = dx_dy_np(a)
    angle = np.arctan2(diff[:, 1], diff[:, 0])
    if as_degrees:
        angle = np.rad2deg(angle)
    return angle


def azim_np(a):
    """Return the azimuth/bearing relative to North

    Requires, `angle_np`, which calls `dx_dy_np` which calls `_new_view_`
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

    Parameters
    ----------
    a : array
        an array of points, derived from a polygon/polyline geometry
    inside : boolean
        determine inside angles, outside if False
    in_deg : bolean
        convert to degrees from radians

    Notes
    -----
    General comments::

        2 points - subtract 2nd and 1st points, effectively making the
        calculation relative to the origin and x axis, aka... slope
        n points - sequential angle between 3 points

    To keep::

        ``keep to convert object to array``
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


def orig_dest_angle(orig, dest, fromNorth=False):
    """Direction of lines formed from an origin to multiple destination points.
    This simply is a shell for `line_dir` in this module.  See the docs there.
    """
    return line_dir(orig, dest, fromNorth)


def line_dir(orig, dest, fromNorth=False):
    """Direction of a line given 2 points, or an origin and multiple
    destinations.

    Parameters
    ----------
    orig, dest : arrays
        2D arrays of representing the start and end coordinates of two point
        lines.
    fromNorth : boolean
        True or False gives angle relative to North or the x-axis.

    Example
    -------
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


# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    print("Script... {}".format(script))
