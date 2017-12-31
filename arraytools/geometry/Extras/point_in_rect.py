# -*- coding: UTF-8 -*-
"""
:Script:   point_in_rect
:Author:   Dan.Patterson@carleton.ca
:Purpose:
:  To determine whether points are within the extents of polygons.
:
:References:
:----------
:  http://stackoverflow.com/questions/30481577/
:       assign-numpy-array-of-points-to-a-2d-square-grid
:  http://stackoverflow.com/questions/33051244/
:       numpy-filter-points-within-bounding-box
:  https://stackoverflow.com/questions/33051244/
:       numpy-filter-points-within-bounding-box/33051576#33051576
:  https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html  ** good
:
:Notes:
:----- loop through all edges of the polygon
: a, ext = array_demo()
: poly = extent_poly(ext)
: p0 = np.array([341999, 5021999])
: p1 = np.mean(poly, axis=0)
: pnts - 10,000 points within the full extent, 401 points within the polygon
:  I like this.,... np.nonzero(x > xs)
:
: (1) pnts_in_extent
:     %timeit pnts_in_extent(pnts, ext, in_out=False)
:     173 µs ± 2.16 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
:
:     %timeit pnts_in_extent(pnts, ext, in_out=True)
:     342 µs ± 9.12 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
:
: (2) pure crossing_num
:     %timeit crossing_num(pnts, poly)
:     369 ms ± 19.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
:
: (3) crossing_num with pnts_in_extent check (current version)
:      %timeit crossing_num(pnts, poly)
:     9.68 ms ± 120 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
:
"""
# ----10| ------20| ------30| ------40| ------50| ------60| ------70| ------80|
import numpy as np
from matplotlib.path import Path
import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

pip = Path.contains_point  # ---- not matplotlib.path.Path.contains_point


def extent_poly(ext):
    """construct the extent rectangle from the extent points which are the
    :  lower left and upper right points [LB, RT]
    """
    LB, RT = ext
    L, B = LB
    R, T = RT
    box = [LB, [L, T], RT, [R, B], LB]
    ext_rect = np.array(box)
    return ext_rect


def pnts_in_extent(pnts, ext, in_out=True):
    """Points in polygon test using numpy and logical_and to find points
    :  within a box/extent.
    :
    :Requires:
    :--------
    :  pnts - an array of points
    :  ext - the extent of the rectangle being tested as an array of the
    :        left bottom (LB) and upper right (RT) coordinates
    :  in_out - boolean, True to return both the inside and outside points.
    :        False for inside only.
    :
    :Notes:
    :-----
    :  comp - np.logical_and( great-eq LB, less RT)  condition check
    :  inside - np.where(np.prod(comp, axis=1) == 1) if both true, product = 1
    :  case - comp returns [True, False] so you take the product
    :  idx_in - indices derived using where since case will be 0 or 1
    :  inside - slice the pnts using idx_in
    """
    outside = None
    LB, RT = ext
    comp = np.logical_and((LB <= pnts), (pnts <= RT))
    case = comp[:, 0] * comp[:, 1]
    idx_in = np.where(case)[0]
    inside = pnts[idx_in]
    if in_out:
        idx_out = np.where(~case)[0]  # invert case
        outside = pnts[idx_out]
    return inside, outside


def crossing_num(pnts, poly):
    """Points in polygon implementation of crossing number largely from pnpoly
    :  in its various incarnations.  This version also does a within extent
    :  test to pre-process the points, keeping those within the extent to be
    :  passed on to the crossing number section.
    :
    :Requires:
    :--------
    : pnts_in_extent - Method to limit the retained points to those within the
    :     polygon extent.  See 'pnts_in_extent' for details
    : pnts - point array
    : poly - polygon, closed-loop as an array
    :
    :Notes:
    :-----
    """
    xs = poly[:, 0]
    ys = poly[:, 1]
    dy = np.diff(ys)
    dx = np.diff(xs)
    ext = np.array([[xs.min(), ys.min()], [xs.max(), ys.max()]])
    inside, outside = pnts_in_extent(pnts, ext, in_out=False)
    is_in = []
    for pnt in inside:
        cn = 0    # the crossing number counter
        x, y = pnt
        for i in range(len(poly)-1):   # edge from V[i] to V[i+1]
            # u = np.logical_and(ys[i] <= y, ys[i+1] > y)  # upward crossing
            # d = np.logical_and(ys[i] >= y, ys[i+1] < y)  # downward crossing
            u = ys[i] <= y < ys[i+1]
            d = ys[i] >= y > ys[i+1]
            if np.logical_or(u, d):       # compute x-coordinate
                vt = (y - ys[i]) / dy[i]
                if x < (xs[i] + vt * dx[i]):
                    cn += 1
        is_in.append(cn % 2)  # either even or odd (0, 1)
    result = inside[np.nonzero(is_in)]
    return result


def pnts_in_extentmpl(pnts, ext):
    """matplotlib incarnation
    :  from matplotlib.path import Path
    :  pip = Path.pnts_in_extent_point
    """
    poly = extent_poly(ext)
    poly = Path(poly)
    is_in = [pip(poly, p) for p in pnts]
    return np.array(is_in)


def array_demo():
    """ used in the testing
    : polygon layers
    : C:\Git_Dan\a_Data\testdata.gdb\Carp_5x5km  full 25 polygons
    : C:\Git_Dan\a_Data\testdata.gdb\subpoly     centre polygon with 'ext'
    : C:\Git_Dan\a_Data\testdata.gdb\centre_4    above, but split into 4
    """
    ext = np.array([[342000, 5022000], [343000, 5023000]])
    in_fc = r'C:\Git_Dan\a_Data\testdata.gdb\xy_10k'
    SR = arcpy.SpatialReference(2951)
    a = arcpy.da.FeatureClassToNumPyArray(in_fc,
                                          ['SHAPE@X', 'SHAPE@Y'],
                                          spatial_reference=SR)
    a0 = a.view(dtype=np.float).reshape(len(a), 2)
    return a0, ext


def src_geom(pnt_fc, poly_fc):
    """ a search cursor approach
    """
    pnts = [p[0] for p in arcpy.da.SearchCursor(pnt_fc, "SHAPE@")]
    polys = [pl[0] for pl in arcpy.da.SearchCursor(poly_fc, "SHAPE@")]
    poly = polys[0]
    is_in = [pnt.within(poly) for pnt in pnts]
    return pnts, poly, is_in


if __name__ == "__main__":
    """make some points for testing, create and extent,
    time each option and optionally graph"""

#    arcpy.env.overwriteOutput = True
#    arcpy.env.workspace = r'C:\Git_Dan\a_Data\testdata.gdb'
#    pnt_fc = r'C:\Git_Dan\a_Data\testdata.gdb\xy_10k'
#    poly_fc = r'C:\Git_Dan\a_Data\testdata.gdb\subpoly'
#    pnts, poly, is_in = src_geom(pnt_fc, poly_fc)
#    pnts_in = sum(is_in)

pnts, ext = array_demo()

poly = extent_poly(ext)
# p0 = np.array([341999, 5021999])
p1 = np.mean(poly, axis=0)

#    arcpy.MakeFeatureLayer_management('xy_10k', 'xy_lyr')
#    # arcpy.MakeFeatureLayer_management('subpoly', 'subpoly_lyr')
#    arcpy.management.SelectLayerByLocation("xy_lyr", "INTERSECT", "subpoly",
#                                           None, "NEW_SELECTION", "NOT_INVERT")
#    matchcount = int(arcpy.GetCount_management('xy_lyr')[0])
#  %timeit pnts_in_extent(a0, ext, True)
#  297 µs ± 2.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# Profiling code
# basic prun
# %prun contains(pnts, in_ext)

# %load_ext line_profiler
# note that it is the function name, then repeated with the parameters
# -f means a function
#
# %lprun -f pnts_in_extent pnts_in_extent(pnts, in_ext)
