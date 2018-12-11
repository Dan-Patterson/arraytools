# -*- coding: UTF-8 -*-
"""
hulls.py
========

Script:   hulls.py

Author:   Dan.Patterson@carleton.ca

Modified: 2018-11-11

Purpose:
--------
Determine convex and concave hulls for point data.  This is a part of the
PointTools toolbox for use in ArcGIS Pro

References:
-----------
concave/convex hulls

`<https://tereshenkov.wordpress.com/2017/11/28/building-concave-hulls-alpha-
shapes-with-pyqt-shapely-and-arcpy/>`_.

`<https://repositorium.sdum.uminho.pt/handle/1822/6429?locale=en>`_.

`<https://community.esri.com/blogs/dan_patterson/2018/03/11/
concave-hulls-the-elusive-container>'_.

`<https://github.com/jsmolka/hull/blob/master/hull.py>'_.

`<https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-
line-segments-intersect#565282>'_.

`<http://www.codeproject.com/Tips/862988/Find-the-intersection-
point-of-two-line-segments>'_.
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import warnings
import math


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

    Requires:
    --------
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


#def knn(pnts, p, k):
#    """
#    Calculates k nearest neighbours for a given point.
#
#    points : array
#        list of points
#    p : two number array-like
#        reference point
#    k : integer
#        amount of neighbours
#    Returns:
#    --------
#    list
#    """
#    s = sorted(pnts,
#               key=lambda x: math.sqrt((x[0]-p[0])**2 + (x[1]-p[1])**2))[0:k]
#    return s


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
        keep = ~np.all(pnts==p, axis=1)
        return pnts[keep]
    #
    def _e_2d_(p, a):
        """ array points to point distance... mini e_dist
        """
        diff = a - p[np.newaxis, :]
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))
    #
    k = max(1, min(abs(int(k)), len(pnts)))
    pnts = _remove_self_(p, pnts)
    d = _e_2d_(p, pnts)
    idx = np.argsort(d)
    if return_dist:
        return pnts[idx][:k], d[idx][:k]
    return pnts[idx][:k]


def concave(points, k):
    """Calculates the concave hull for given points

    Requires:
    --------
    points : array-like
        initially the input set of points with duplicates removes and
        sorted on the Y value first, lowest Y at the top (?)
    k : integer
        initially the number of points to start forming the concave hull,
        k will be the initial set of neighbors

    Notes:
    ------
    This recursively calls itself to check concave hull.

    p_set : The working copy of the input points
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
            p_set.append(frst_p)         # Add first point again
        knn_pnts = knn(cur_p, p_set, k)  # Find nearest neighbours
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


def nd_intersect2(a, b, indices=False):
    """Intersection of 2, nx2 arrays using views.  Return optional indices

    `<https://stackoverflow.com/questions/9269681/intersection-of-2d-
    numpy-ndarrays>`_.
    """
    a_view = a.view([('',a.dtype)]*a.shape[1])
    b_view = b.view([('',b.dtype)]*b.shape[1])
    if indices:
        ab, idx0, idx1 = np.intersect1d(a_view, b_view, return_indices=indices)
        return ab.view(a.dtype).reshape(-1, a.shape[1]), idx0, idx1
    else:
        ab = np.intersect1d(a_view, b_view)
        return ab.view(a.dtype).reshape(-1, a.shape[1])

def nd_intersect(a, b, invert=False):
    """Intersect of two, 2D arrays using views and in1d

    Parameters:
    -----------
    a, b : arrays
        Arrays are assumed to have a shape = (N, 2)

    `<https://github.com/numpy/numpy/blob/master/numpy/lib/arraysetops.py>`_.

    `<https://stackoverflow.com/questions/9269681/intersection-of-2d-
    numpy-ndarrays>`_.
    """
    a_view = a.view([('',a.dtype)]*a.shape[1])
    b_view = b.view([('',b.dtype)]*b.shape[1])
    if len(a) < len(b):
        idx = np.in1d(a_view, b_view, assume_unique=False, invert=False)
        return a[idx] 
    else:
        idx = np.in1d(b_view, a_view, assume_unique=False, invert=False)
        return b[idx] 

#  https://github.com/numpy/numpy/blob/master/numpy/lib/arraysetops.py

def nd_difference(a, b):
    """Difference of 2, nx2 arrays using views.  Try to have array `a` the 
    smaller of the two arrays... not necessary, but less checks.  Returns the
    difference between the two arrays.

    Parameters:
    -----------
    a, b : arrays
        Arrays are assumed to have a shape = (N, 2)
    """
    a_view = a.view([('',a.dtype)]*a.shape[1])
    b_view = b.view([('',b.dtype)]*b.shape[1])
    if len(a) < len(b):
        idx = np.in1d(a_view, b_view, assume_unique=False, invert=True)
        return a[idx]
    else:
        idx = np.in1d(b_view, a_view, assume_unique=False, invert=True)
        return b[idx]

def nd_diffxor(a, b, uni=False):
    """using setxor... it is slower than nd_diff, 36 microseconds vs 18.2
    but this is faster for large sets
    """
    a_view = a.view([('',a.dtype)]*a.shape[1])
    b_view = b.view([('',b.dtype)]*b.shape[1])
    ab = np.setxor1d(a_view, b_view, assume_unique=uni)
    return ab.view(a.dtype).reshape(-1, a.shape[1])


# ----------------------------------------------------------------------
# .... running script or testing code section

def ice():
    """Ice demo for concave hull generation"""
    pth = r"C:\GIS\A_Tools_scripts\Polygon_lineTools\Data\pointset.csv"
    a = np.loadtxt(pth, delimiter=",", skiprows=1)
    return a


def c_():
    """Letter c for concave hull determination
    """
    c = np.array([[ 0, 0], [ 0, 100], [100, 100], [100,  80], [ 20,  80],
                  [ 20, 20], [100, 20], [100, 0], [ 0, 0]])
    return c


def _demo():
    """Demo data
    """
    hull_type = 'concave'
    k_factor = 3 # 3 to 11
    # (1) ---- get the points
    #a = np.array([[0, 0], [0, 10], [10, 0], [0,0]])

    a = np.array([[0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                  [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                  [1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
                  [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                  [1, 1, 0, 1, 1, 1, 0, 1, 0, 0],
                  [0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                  [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]])
    xs, ys = np.where(a == 1)
    cell_size =  10
    xy = np.array(list(zip(xs*cell_size, ys*cell_size)))
    # (2) ---- for each group, perform the concave hull
    in_arrays = [a]
    groups = [xy] #[a[np.where(a[group_by] == i)[0]] for i in uniq]
    hulls = []
    out_arrs = []
    for i in range(0, len(groups)):
        p = groups[i]
        # ---- point preparation section ------------------------------------
        p = np.array(list(set([tuple(i) for i in p])))  # Remove duplicates
        idx_cr = np.lexsort((p[:, 0], p[:, 1]))         # indices of sorted array
        in_pnts = np.asarray([p[i] for i in idx_cr])    # p[idx_cr]  #
        in_pnts = in_pnts.tolist()
        in_pnts = [tuple(i) for i in in_pnts]
        if hull_type == 'concave':
            cx = np.array(concave(in_pnts, k_factor))  # requires a list of tuples
        else:
            cx = np.array(convex(in_pnts))
        hulls.append(cx)
        z = np.zeros_like(in_arrays[i])
        for i in cx:
            x = int(i[0]//cell_size)
            y = int(i[1]//cell_size)
            z[x, y] = 1
        out_arrs.append(z)
    return a, xy, hulls, out_arrs
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
