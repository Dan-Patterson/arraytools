# -*- coding: UTF-8 -*-
"""
geom
====

Script :   geom.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-10-15

Purpose :  tools for working with numpy arrays

Functions: see __all__ for a complete listing

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

------
"""
# ---- imports, formats, constants ----
import sys
from textwrap import dedent
import numpy as np
from numpy.lib.stride_tricks import as_strided
# from arraytools.fc import _xy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.2f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=3, suppress=True,
                    nanstr='nan', infstr='inf',
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['_flat_', '_unpack', 'segment', 'stride',
           '_new_view_', '_view_', '_reshape_',
           '_min', '_max', '_extent',
           '_center', '_centroid', 'centers', 'centroids',
           'e_area', 'e_dist', 'e_leng',
           'areas', 'lengths',
           'total_length', 'seg_lengths',
           'radial_sort',
           'dx_dy_np', 'angle_np', 'azim_np',
           'angle_2pnts', 'angle_seq', 'angles_poly', 'dist_bearing',
           '_densify_2D', '_convert', 'densify',
           'simplify',
           'rotate',  'trans_rot', 'repeat',
           'circle', 'ellipse',
           'rectangle', 'hex_flat', 'hex_pointy'
           ]
"""
['__all__', '__builtins__', '__cached__', '__doc__', '__file__',
 '__loader__', '__name__', '__package__', '__spec__', '_arrs_', '_center',
 '_centroid', '_convert', '_demo', '_densify_2D', '_extent', '_flat_',
 '_max', '_min', '_new_view_', '_reshape_', '_test', '_unpack', '_view_',
 'adjacency_edge', 'angle_2pnts', 'angle_between', 'angle_np', 'angle_seq',
 'angles_poly', 'areas', 'as_strided', 'azim_np', 'centers', 'centroids',
 'circle', 'dedent', 'densify', 'dist_bearing', 'dx_dy_np', 'e_area',
 'e_dist', 'e_leng', 'ellipse', 'ft', 'hex_flat', 'hex_pointy',
 'intersect_pnt', 'lengths', 'np', 'pnt_', 'radial_sort', 'rectangle',
 'repeat', 'rotate', 'script', 'seg_lengths', 'segment', 'simplify',
 'stride', 'sys', 'total_length', 'trans_rot']"""

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
        if hasattr(x, '__iter__'):
            xy.extend(_unpack(x))
        else:
            xy.append(x)
    return xy


def segment(a):
    """Segment poly* structures into o-d pairs from start to finish

    `a` : array
        A 2D array of x,y coordinates representing polyline or polygons.
    `fr_to` : array
        Returns a 3D array of point pairs.
    """
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
    if (len(a.dtype) > 1):
        shp = a.shape[0]
        a = a.view(dtype='float64')
        a = a.reshape(shp, 2)
    return a


def _view_(a):
    """Return a view of the array using the dtype and length"""
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
    if not isinstance(a, (np.ndarray)):
        a = np.asanyarray(a)
    shp = len(a.shape)
    _len = len(a.dtype)
    if a.dtype.kind == 'O':
        if len(a[0].shape) == 1:
            return np.asarray([_view_(i) for i in a])
        else:
            return _view_(a)
    if _len == 0:
        if shp == 1:
            return [_view_(i) for i in a]
        elif shp == 2:
            return a
        elif shp > 2:
            tmp = a.reshape(np.product(a.shape[:-1]), 2)
            return tmp.view('<f8')
    elif _len == 1:
        fld_name = a.dtype.names[0]  # assumes 'Shape' field is the geometry
        return a[fld_name]
    elif _len >= 2:
        if shp == 1:
            if len(a) == a.shape[0]:
                a = _view_(a)
            else:
                a = np.asanyarray([_view_(i) for i in a])
        else:
            a = np.asanyarray([_view_(i) for i in a])
        return a
    else:
        return a


# ---- extent, mins and maxs ------------------------------------------------
def _min(a):
    """Array minimums
    """
    a = _reshape_(a)
    if (a.dtype.kind == 'O') or (len(a.shape) > 2):
        mins = np.asanyarray([i.min(axis=0) for i in a])
    else:
        mins = a.min(axis=0)
    return mins


def _max(a):
    """Array maximums
    """
    a = _reshape_(a)
    if (a.dtype.kind == 'O') or (len(a.shape) > 2):
        maxs = np.asanyarray([i.max(axis=0) for i in a])
    else:
        maxs = a.max(axis=0)
    return maxs


def _extent(a):
    """Array extent values
    """
    a = _reshape_(a)
    if isinstance(a, (list, tuple)):
        a = np.asanyarray(a)
    if (a.dtype.kind == 'O') or (len(a.shape) > 2):
        mins = _min(a)
        maxs = _max(a)
        return np.hstack((mins, maxs))
    else:
        L, B = _min(a)
        R, T = _max(a)
        return np.asarray([L, B, R, T])


# ---- centers --------------------------------------------------------------
def _center(a, remove_dup=True):
    """Return the center of an array. If the array represents a polygon, then
    a check is made for the duplicate first and last point to remove one.
    """
    if remove_dup:
        if np.all(a[0] == a[-1]):
            a = a[:-1]
    return a.mean(axis=0)


def _centroid(a, a_6=None):
    """Return the centroid of a closed polygon.

    `a` : array
        A 2D or more of point coordinates
    `a_6` : number
        If area has been precalculated, you can use its value.
    `e_area` : function (required)
        Contained in this module.
    """
    x, y = a.T
    t = ((x[:-1] * y[1:]) - (y[:-1] * x[1:]))
    if a_6 is None:
        a_6 = e_area(a) * 6.0  # area * 6.0
    x_c = np.sum((x[:-1] + x[1:]) * t) / a_6
    y_c = np.sum((y[:-1] + y[1:]) * t) / a_6
    return np.asarray([-x_c, -y_c])


def centroids(a, remove_dup=True):
    """batch centroids (ie _centroid)
    """
    a = np.asarray(a)
    if a.dtype == 'O':
        tmp = [_reshape_(i) for i in a]
        return np.asarray([_centroid(i) for i in tmp])
    if len(a.dtype) >= 1:
        a = _reshape_(a)
    if a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    c = np.asarray([_centroid(i) for i in a]).squeeze()
    return c


def centers(a, remove_dup=True):
    """batch centres (ie _center)
    """
    a = np.asarray(a)
    if a.dtype == 'O':
        tmp = [_reshape_(i) for i in a]
        return np.asarray([_center(i, remove_dup) for i in tmp])
    if len(a.dtype) >= 1:
        a = _reshape_(a)
    if a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    c = np.asarray([_center(i, remove_dup) for i in a]).squeeze()
    return c


# ---- distance, length and area --------------------------------------------
# ----
def e_area(a, b=None):
    """Area calculation, using einsum.

    Some may consider this overkill, but consider a huge list of polygons,
    many multipart, many with holes and even multiple version therein.

    Requires:
    --------
    `a` : array
        Either a 2D+ array of coordinates or arrays of x, y values
    `b` : array, optional
        If a < 2D, then the y values need to be supplied
    Outer rings are ordered clockwise, inner holes are counter-clockwise

    Notes:
    -----
        See ein_geom.py for examples

    """
    a = np.asanyarray(a)
    if b is None:
        xs = a[..., 0]
        ys = a[..., 1]
    else:
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

    `a`, `b` : array like
        Inputs, list, tuple, array in 1, 2 or 3D form
    `metric` : string
        euclidean ('e', 'eu'...), sqeuclidean ('s', 'sq'...),
    -----------------------------------------------------------------------
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
    """Calls e_leng to calculate lengths for many types of nested objects.
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
def radial_sort(pnts, cent=None):
    """Sort about the point cloud center or from a given point

    `pnts` : points
        An array of points (x,y) as array or list
    `cent` : coordinate
        list, tuple, array of the center's x,y coordinates
    >>> cent = [0, 0] or np.array([0, 0])

    Returns:
    -------
        The angles in the range -180, 180 x-axis oriented

    """
    pnts = np.asarray(pnts, dtype=np.float64)
    if cent is None:
        cent = _center(pnts, remove_dup=False)
    ba = pnts - cent
    ang_ab = np.arctan2(ba[:, 1], ba[:, 0])
    ang_ab = np.degrees(ang_ab)
    sort_order = np.argsort(ang_ab)
    return ang_ab, sort_order


# ---- angle related functions ----------------------------------------------
# ---- structured arrays
def dx_dy_np(a, fld):
    """Sequential difference in the x/y pairs from a table/array
    :
    """
    out = np.zeros(a[fld].shape)
    diff = a[fld][1:] - a[fld][:-1]
    out[1:] = diff
    return out


def angle_np(a, fld, as_degrees=True):
    """Angle between successive points.
       Requires dx_dy_np
    """
    diff = dx_dy_np(a, fld)
    angle = np.arctan2(diff[:, 1], diff[:, 0])
    if as_degrees:
        angle = np.rad2deg(angle)
    return angle


def azim_np(a, fld):
    """Return the azimuth/bearing relative to North"""
    s = angle_np(a, fld, as_degrees=True)
    azim = np.where(s <= 0, 90. - s,
                    np.where(s > 90, 450.0 - s, 90.0 - s))
    return azim


# ---- ndarrays
def angle_2pnts(p0, p1):
    """Two point angle. p0 represents the `from` point and p1 the `to` point.
    ::
        angle = atan2(vector2.y, vector2.x) - atan2(vector1.y, vector1.x)
    Accepted answer from the poly_angles link
    """
    p0, p1 = [np.asarray(i) for i in [p0, p1]]
    ba = p1 - p0
    ang_ab = np.arctan2(*ba[::-1])
    return np.rad2deg(ang_ab % (2 * np.pi))


def angle_seq(a):
    """Sequential angles for a points list
    ::
        angle = atan2(vector2.y, vector2.x) - atan2(vector1.y, vector1.x)
    Accepted answer from the poly_angles link
    """
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

    Sample data

    >>>

    """
    if len(a) < 2:
        return None
    elif len(a) == 2:
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
        ang = np.where(ang<0, ang + two_pi, ang)
    else:
        ang = np.where(ang>0, two_pi - ang, ang)
    if in_deg:
        angles = np.degrees(ang)
    return angles


def dist_bearing(orig=(0, 0), bearings=None, dists=None, prn=False):
    """Point locations given distance and bearing.
    Now only distance and angle are known.  Calculate the point coordinates
    from distance and angle

    References:
    ----------
    1. https://community.esri.com/thread/66222

    2. https://community.esri.com/blogs/dan_patterson/2018/01/21/ \
    origin-distances-and-bearings-geometry-wanderings

    Notes:
    -----
    Sample calculation
    ::
          bearings = np.arange(0, 361, 10.)  # 37 bearings
          dists = np.random.randint(10, 500, len(bearings)) * 1.0
          dists = np.ones((len(bearings),))
          dists.fill(100.)
          data = dist_bearing(orig=orig, bearings=bearings, dists=dists)

    Create a featureclass from the results
    ::
       shapeXY = ['X_f', 'Y_f']
       fc_name = r"C:\path\Geodatabase.gdb\featureclassname"
       arcpy.da.NumPyArrayToFeatureClass(out, fc_name, ['Xn', 'Yn'], "2951")
       # ... syntax
       arcpy.da.NumPyArrayToFeatureClass(
                          in_array=out, out_table=fc_name,
                          shape_fields=shapeXY, spatial_reference=SR)
    """
    orig = np.array(orig)
    rads = np.deg2rad(bearings)
    dx = np.sin(rads) * dists
    dy = np.cos(rads) * dists
    x_t = np.cumsum(dx) + orig[0]
    y_t = np.cumsum(dy) + orig[1]
    xy_f = np.array(list(zip(x_t[:-1], y_t[:-1])))
    xy_f = np.vstack((orig, xy_f))
    stack = (xy_f[:, 0], xy_f[:, 1], x_t, y_t, dx, dy, dists, bearings)
    data = np.vstack(stack).T
    names = ['X_f', 'Y_f', "X_t", "Yt", "dx", "dy", "dist", "bearing"]
    N = len(names)
    if prn:  # ---- just print the results ----------------------------------
        frmt = "Origin (0,0)\n" + "{:>10s}"*N
        print(frmt.format(*names))
        frmt = "{: 10.2f}"*N
        for i in data:
            print(frmt.format(*i))
        return data
    else:  # ---- produce a structured array from the output ----------------
        names = ", ".join(names)
        kind = ["<f8"]*N
        kind = ", ".join(kind)
        out = data.transpose()
        out = np.core.records.fromarrays(out, names=names, formats=kind)
        return out


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
            for i in range(len(p)):
                pp = p[i]
                shp = _densify_2D(pp, fact=fact)
                arc_pnts = [Point(*p) for p in shp]
                sub_out.append(arc_pnts)
            out.append(sub_out)
    return out


def densify(polys, fact=2, sp_ref=None):
    """Convert polygon objects to arrays, densify.

    Requires:
    --------
        `_densify_2D` - the function that is called for each shape part

        `_unpack` - unpack objects
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
        return np.array([np.nan, np.nan])
    elif isinstance(p, (np.ndarray, list, tuple)):
        if len(p) == 2:
            p = np.array([p[0], p[1]])
        else:
            p = np.array([p[0], p[0]])
        return p


def rotate(pnts, angle=0):
    """rotate points about the origin in degrees, (+ve for clockwise) """
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
    unique :
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
    R = np.array(((c, s), (-s,  c)))
    return  np.einsum('ij,kj->ik', a - cent, R) + cent


def repeat(seed=None, corner=[0, 0], cols=1, rows=1, angle=0):
    """Create the array of pnts to pass on to arcpy using numpy magic to
    produce a fishnet of the desired in_shp.

    `seed` : array
        Use grid_array, hex_flat or hex_pointy.  You specify the width
        and height or its ratio when making the shapes
    `corner` : point coordinates
        The lower left corner of the shape pattern
    `rows`, `cols` : ints
        The number of rows and columns to produce
    `angle` : number
        Rotation angle in degrees
    """
    if seed is None:
        a = rectangle(dx=1, dy=1, cols=3, rows=3)
    else:
        a = np.asarray(seed)
    if angle != 0:
        a = [rotate(p, angle) for p in a]      # rotate the scaled points
    pnts = [p + corner for p in a]            # translate them
    return pnts


def circle(radius=1.0, theta=10.0, xc=0.0, yc=0.0):
    """Produce a circle/ellipse depending on parameters.

    `radius` : number
        Distance from centre
    `theta` : number
        Angle of densification of the shape around 360 degrees
    """
    angles = np.deg2rad(np.arange(180.0, -180.0-theta, step=-theta))
    x_s = radius*np.cos(angles) + xc    # X values
    y_s = radius*np.sin(angles) + yc    # Y values
    pnts = np.c_[x_s, y_s]
    return pnts


def ellipse(x_radius=1.0, y_radius=1.0,  theta=10., xc=0.0, yc=0.0):
    """Produce an ellipse depending on parameters.

    `radius` : number
        Distance from centre in the X and Y directions
    `theta` : number
        Angle of densification of the shape around 360 degrees
    """
    angles = np.deg2rad(np.arange(180.0, -180.0-theta, step=-theta))
    x_s = x_radius*np.cos(angles) + xc    # X values
    y_s = y_radius*np.sin(angles) + yc    # Y values
    pnts = np.c_[x_s, y_s]
    return pnts


def rectangle(dx=1, dy=1, cols=1, rows=1):
    """Create the array of pnts to pass on to arcpy using numpy magic

    `dx` : number
        Increment in x direction, +ve moves west to east, left/right
    `dy` : number
        Increment in y direction, -ve moves north to south, top/bottom
    `rows`, `cols` : ints
        Row and columns to produce
    """
    X = [0.0, 0.0, dx, dx, 0.0]       # X, Y values for a unit square
    Y = [0.0, dy, dy, 0.0, 0.0]
    seed = np.array(list(zip(X, Y)))  # [dx0, dy0] keep for insets
    a = [seed + [j * dx, i * dy]       # make the shapes
         for i in range(0, rows)   # cycle through the rows
         for j in range(0, cols)]  # cycle through the columns
    a = np.asarray(a)
    return a


def hex_flat(dx=1, dy=1, cols=1, rows=1):
    """Generate the points for the flat-headed hexagon

    `dy_dx` : number
        The radius width, remember this when setting hex spacing
    `dx` : number
        Increment in x direction, +ve moves west to east, left/right
    `dy` : number
        Increment in y direction, -ve moves north to south, top/bottom
    """
    f_rad = np.deg2rad([180., 120., 60., 0., -60., -120., -180.])
    X = np.cos(f_rad) * dy
    Y = np.sin(f_rad) * dy            # scaled hexagon about 0, 0
    seed = np.array(list(zip(X, Y)))  # array of coordinates
    dx = dx * 1.5
    dy = dy * np.sqrt(3.)/2.0
    hexs = [seed + [dx * i, dy * (i % 2)] for i in range(0, cols)]
    m = len(hexs)
    for j in range(1, rows):  # create the other rows
        hexs += [hexs[h] + [0, dy * 2 * j] for h in range(m)]
    return hexs


def hex_pointy(dx=1, dy=1, cols=1, rows=1):
    """Pointy hex angles, convert to sin, cos, zip and send

    `dy_dx` - number
        The radius width, remember this when setting hex spacing
    `dx` : number
        Increment in x direction, +ve moves west to east, left/right
    `dy` : number
        Increment in y direction, -ve moves north to south, top/bottom
    """
    p_rad = np.deg2rad([150., 90, 30., -30., -90., -150., 150.])
    X = np.cos(p_rad) * dx
    Y = np.sin(p_rad) * dy      # scaled hexagon about 0, 0
    seed = np.array(list(zip(X, Y)))
    dx = dx * np.sqrt(3.)/2.0
    dy = dy * 1.5
    hexs = [seed + [dx * i * 2, 0] for i in range(0, cols)]
    m = len(hexs)
    for j in range(1, rows):  # create the other rows
        hexs += [hexs[h] + [dx * (j % 2), dy * j] for h in range(m)]
    return hexs


# ---- Extras ----------------------------------------------------------------
#
def _test(a0=None):
    """testing stuff using a0"""
    import math
    if a0 is None:
        a0 = np.array([[10, 10], [10, 20], [20, 20], [10, 10]])
    x0, y0 = p0 = a0[-2]
    x1, y1 = p1 = a0[-1]
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


def adjacency_edge():
    """keep... adjacency-edge list association
    : the columns are 'from', the rows are 'to' the cell value is the 'weight'.
    """
    adj = np.random.randint(1, 4, size=(5, 5))
    r, c = adj.shape
    mask = ~np.eye(r, c, dtype='bool')
    dt = [('Source', '<i4'), ('Target', '<i4'), ('Weight', '<f8')]
    m = mask.ravel()
    XX, YY = np.meshgrid(np.arange(c), np.arange(r))
    mmm = np.stack((m, m, m), axis=1)
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


def _arrs_(prn=True):
    """Sample arrays to test various cases
    """
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


def intersect_pnt(p0, p1, p2, p3):
    """Returns the point of intersection of the segment passing through two
    line segments (p0, p1) and (p2, p3)

    Notes:
    ------

    >>> p0, p1, p2, p3
    (array([0, 0]), array([10, 10]),array([0, 5]), array([5, 0]))
     s: array([[ 0,  0],    h: array([[  0.,   0.,   1.],
               [10, 10],              [ 10.,  10.,   1.],
               [ 0,  5],              [  0.,   5.,   1.],
               [ 5,  0]])             [  5.,   0.,   1.]])

    Reference:
    ---------

    [1]
    https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    """
    s = np.vstack([p0, p1, p2, p3])      # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])            # get first line
    l2 = np.cross(h[2], h[3])            # get second line
    x, y, z = np.cross(l1, l2)           # point of intersection
    if z == 0:                           # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    from _common import fc_info
#    from fc import _xyID, obj_array, _two_arrays
#    from tools import group_pnts

#    args = _arrs_(prn=False)  # prn=True to see array properties
#    a0, a1, a2, a3, a_1a, a_1b, a_2a, a_2b, a_3a, a_3b = args[:10]
#    sze, shp, dtn = args[10:]
#    a = np.array([[0, 0.05], [1, 1.05], [2, 1.95], [3, 3.0],
#                  [4, 4.1], [5, 5.2], [6, 5.9]])
#    dist, xc, yc, pc, slope, xn, yn, x2 = _test(a0)
#    fc = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb\polylines_pnts"

# for Ontario_LCC
# total_length(a)  #: 6804096.2018476073  same as arcmap
# areas(a)  #: [1074121438784.0]  1074121438405.34021

#     ---- end
#    in_fc = r'C:\Git_Dan\a_Data\arcpytools_demo.gdb\Can_geom_sp_LCC'
#    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\Carp_5x5"   # full 25 polygons
#    a = _xy(in_fc)
#    a_s = group_pnts(a, key_fld='IDs', shp_flds=['Xs', 'Ys'])
#    a_s = np.asarray(a_s)
#    a_area = areas(a_s)
#    a_tot_leng = total_length(a_s)
#    a_seg_leng = seg_lengths(a_s)
#    a_ng = angles_poly(a1)
#    v = r'C:\Git_Dan\arraytools\Data\sample_100K.npy'  # 20, 1000, 10k, 100K
#    oa = obj_array(in_fc)
#    ta = _two_arrays(in_fc, both=True, split=True)
