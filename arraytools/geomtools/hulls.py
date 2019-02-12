# -*- coding: UTF-8 -*-
"""
========
hulls.py
========

Script :  hulls.py

Author :  Dan.Patterson@carleton.ca

Modified : 2019-02-11

Notes
-----
Determine convex and concave hulls for point data.  This is a part of the
arraytools tools and is used in the PointTools toolbox for use in ArcGIS Pro

References
----------
concave/convex hulls

`<https://www.researchgate.net/publication/220868874_Concave_hull_A_k-
nearest_neighbours_approach_for_the_computation_of_the_region_occupied_by_a
_set_of_points>`_.

`<https://tereshenkov.wordpress.com/2017/11/28/building-concave-hulls-alpha-
shapes-with-pyqt-shapely-and-arcpy/>`_.

`<https://repositorium.sdum.uminho.pt/handle/1822/6429?locale=en>`_.

`<https://community.esri.com/blogs/dan_patterson/2018/03/11/
concave-hulls-the-elusive-container>`_.

`<https://github.com/jsmolka/hull/blob/master/hull.py>`_.

`<https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-
line-segments-intersect#565282>`_.

`<http://www.codeproject.com/Tips/862988/Find-the-intersection-
point-of-two-line-segments>`_.

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

# ---- imports, formats, constants ----
import sys
import math
import warnings
import numpy as np

warnings.simplefilter('ignore', FutureWarning)

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

PI = math.pi


def pnt_in_list(pnt, pnts_list):
    """Check to see if a point is in a list of points
    """
    is_in = np.any([np.isclose(pnt, i) for i in pnts_list])
    return is_in


def intersects(*args):
    """Line intersection check.  Two lines or 4 points that form the lines.

    Parameters
    ----------
      intersects(line0, line1) or intersects(p0, p1, p2, p3)
        p0, p1 -> line 1
        p2, p3 -> line 2

    Returns:
    --------
        boolean, if the segments do intersect

    References:
    -----------
    `<https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-
    line-segments-intersect#565282>`_.

    """
    if len(args) == 2:
        p0, p1, p2, p3 = *args[0], *args[1]
    elif len(args) == 4:
        p0, p1, p2, p3 = args
    else:
        raise AttributeError("Pass 2, 2-pnt lines or 4 points to the function")
    #
    # ---- First check ----   np.cross(p1-p0, p3-p2 )
    p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y = *p0, *p1, *p2, *p3
    s10_x = p1_x - p0_x
    s10_y = p1_y - p0_y
    s32_x = p3_x - p2_x
    s32_y = p3_y - p2_y
    denom = s10_x * s32_y - s32_x * s10_y
    if denom == 0.0:
        return False
    #
    # ---- Second check ----  np.cross(p1-p0, p0-p2 )
    den_gt0 = denom > 0
    s02_x = p0_x - p2_x
    s02_y = p0_y - p2_y
    s_numer = s10_x * s02_y - s10_y * s02_x
    if (s_numer < 0) == den_gt0:
        return False
    #
    # ---- Third check ----  np.cross(p3-p2, p0-p2)
    t_numer = s32_x * s02_y - s32_y * s02_x
    if (t_numer < 0) == den_gt0:
        return False
    #
    if ((s_numer > denom) == den_gt0) or ((t_numer > denom) == den_gt0):
        return False
    #
    # ---- check to see if the intersection point is one of the input points
    t = t_numer / denom
    # substitute p0 in the equation
    x = p0_x + (t * s10_x)
    y = p0_y + (t * s10_y)
    # be careful that you are comparing tuples to tuples, lists to lists
    if sum([(x, y) == tuple(i) for i in [p0, p1, p2, p3]]) > 0:
        return False
    return True


def angle(p0, p1, prv_ang=0):
    """Angle between two points and the previous angle, or zero.
    """
    ang = math.atan2(p0[1] - p1[1], p0[0] - p1[0])
    a0 = (ang - prv_ang)
    a0 = a0 % (PI * 2) - PI
    return a0


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
        return np.einsum('ij,ij->i', diff, diff)
    #
    p = np.asarray(p)
    k = max(1, min(abs(int(k)), len(pnts)))
    pnts = _remove_self_(p, pnts)
    d = _e_2d_(p, pnts)
    idx = np.argsort(d)
    if return_dist:
        return pnts[idx][:k], d[idx][:k]
    return pnts[idx][:k]


def knn0(pnts, p, k):
    """Calculates `k` nearest neighbours for a given point, `p`, relative to
     otherpoints.

    Parameters
    ----------
    points : array
        list of points
    p : array-like
        reference point, two numbers representing x, y
    k : integer
        number of neighbours

    Returns
    -------
    list of the k nearest neighbours, based on squared distance
    """
    p = np.asarray(p)
    pnts = np.asarray(pnts)
    diff = pnts - p[np.newaxis, :]
    d = np.einsum('ij,ij->i', diff, diff)
    idx = np.argsort(d)[:k]
#    s = [i.tolist() for i in pnts[idx]]
    return pnts[idx].tolist()


def find_a_in_b(a, b, a_fields=None, b_fields=None):
    """Find the indices of the elements in a smaller 2d array contained in
    a larger 2d array. If the arrays are stuctured with field names,then these
    need to be specified.  It should go without saying that the dtypes need to
    be the same.

    Parameters
    ----------
    a, b : 1D or 2D, ndarray or structured/record arrays
        The arrays are arranged so that `a` is the smallest and `b` is the
        largest.  If the arrays are stuctured with field names, then these
        need to be specified.  It should go without saying that the dtypes
        need to be the same.
    a_fields, b_fields : list of field names
        If the dtype has names, specify these in a list.  Both do not need
        names.

    Examples
    --------
    To demonstrate, a small array was made from the last 10 records of a larger
    array to check that they could be found.

    >>> a.dtype
    ``([('ID', '<i4'), ('X', '<f8'), ('Y', '<f8'), ('Z', '<f8')])``
    >>> b.dtype ``([('X', '<f8'), ('Y', '<f8')])``
    >>> a.shape, b.shape # ((69688,), (10,))
    >>> find_a_in_b(a, b, flds, flds)
    array([69678, 69679, 69680, 69681, 69682,
           69683, 69684, 69685, 69686, 69687], dtype=int64)

    References
    ----------
    This is a function from the arraytools.tbl module
    `<https://stackoverflow.com/questions/38674027/find-the-row-indexes-of-
    several-values-in-a-numpy-array/38674038#38674038>`_.
    """
    def _view_(a):
        """from the same name in arraytools"""
        return a.view((a.dtype[0], len(a.dtype.names)))
    #
    small, big = [a, b]
    if a.size > b.size:
        small, big = [b, a]
    if a_fields is not None:
        small = small[a_fields]
        small = _view_(small)
    if b_fields is not None:
        big = big[b_fields]
        big = _view_(big)
    if a.ndim >= 1:  # last slice, if  [:2] instead, it returns both indices
        indices = np.where((big == small[:, None]).all(-1))[1]
    return indices


def concave(points, k, pip_check=False):
    """Calculates the concave hull for given points

    Parameters
    ----------
    points : array-like
        initially the input set of points with duplicates removes and
        sorted on the Y value first, lowest Y at the top (?)
    k : integer
        initially the number of points to start forming the concave hull,
        k will be the initial set of neighbors
    pip_check : boolean
        Whether to do the final point in polygon check.  Not needed for closely
        spaced dense point patterns.
    knn0, intersects, angle, point_in_polygon : functions
        Functions used by `concave`

    Notes:
    ------
    This recursively calls itself to check concave hull.

    p_set : The working copy of the input points

    70,000 points with final pop check removed, 1011 pnts on ch
        23.1 s ± 1.13 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
        2min 15s ± 2.69 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """
    k = max(k, 3)  # Make sure k >= 3
    if isinstance(points, np.ndarray):  # Remove duplicates if not done already
        p_set = np.unique(points, axis=0).tolist()
    else:
        pts = []
        p_set = [pts.append(i) for i in points if i not in pts] # Remove duplicates
        p_set = np.array(p_set)
        del pts
    if len(p_set) < 3:
        raise Exception("p_set length cannot be smaller than 3")
    elif len(p_set) == 3:
        return p_set  # Points are a polygon already
    k = min(k, len(p_set) - 1)  # Make sure k neighbours can be found

    frst_p = cur_p = min(p_set, key=lambda x: x[1])
    hull = [frst_p]       # Initialize hull with first point
    p_set.remove(frst_p)  # Remove first point from p_set
    prev_ang = 0

    while (cur_p != frst_p or len(hull) == 1) and len(p_set) > 0:
        if len(hull) == 3:
            p_set.append(frst_p)          # Add first point again
        knn_pnts = knn0(p_set, cur_p, k)  # Find nearest neighbours
        cur_pnts = sorted(knn_pnts, key=lambda x: -angle(x, cur_p, prev_ang))
        its = True
        i = -1
        while its and i < len(cur_pnts) - 1:
            i += 1
            last_point = 1 if cur_pnts[i] == frst_p else 0
            j = 1
            its = False
            while not its and j < len(hull) - last_point:
                its = intersects(hull[-1], cur_pnts[i], hull[-j - 1], hull[-j])
                j += 1
        if its:  # All points intersect, try a higher number of neighbours
            return concave(points, k + 1)
        prev_ang = angle(cur_pnts[i], cur_p)
        cur_p = cur_pnts[i]
        hull.append(cur_p)  # Valid candidate was found
        p_set.remove(cur_p)
    if pip_check:
        for point in p_set:
            if not point_in_polygon(point, hull):
                return concave(points, k + 1)
    #
    hull = np.array(hull)
    return hull


# ---- convex hull ----------------------------------------------------------
#
def cross(o, a, b):
    """Cross-product for vectors o-a and o-b
    """
    xo, yo = o
    xa, ya = a
    xb, yb = b
    return (xa - xo)*(yb - yo) - (ya - yo)*(xb - xo)
#    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex(points):
    """Calculates the convex hull for given points
    :Input is a list of 2D points [(x, y), ...]
    """
    if isinstance(points, np.ndarray):
        points = np.unique(points, axis=0)
    else:
        pts = []
        points = [pts.append(i) for i in points if i not in pts] # Remove duplicates
        del pts
    if len(points) <= 1:
        return points
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    #print("lower\n{}\nupper\n{}".format(lower, upper))
    return np.array(lower[:-1] + upper)  # upper[:-1]) # for open loop



# ----------------------------------------------------------------------
# .... running script or testing code section


def c_():
    """Letter c for concave hull determination
    """
    c = np.array([[0, 0], [0, 100], [100, 100], [100, 80], [20, 80],
                  [20, 20], [100, 20], [100, 0], [0, 0]])
    return c


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
    print(script)
