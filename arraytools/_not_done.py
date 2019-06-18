# -*- coding: utf-8 -*-
"""

=======

Script :   .py

Author :   Dan_Patterson@carleton.ca

Modified : 2019-

Purpose :  Tools for

Notes:

References:

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import numpy as np
#from scipy.spatial import cKDTree, KDTree
#from arraytools.geomtools import hulls

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=3, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

def isPointInPath(x, y, poly):
    """
    x, y -- x and y coordinates of point
    poly -- a list of tuples [(x, y), (x, y), ...]

    `<https://en.wikipedia.org/wiki/Even–odd_rule>`_.
    """
    num = len(poly)
    i = 0
    j = num - 1
    c = False
    for i in range(num):
        if ((poly[i][1] > y) != (poly[j][1] > y)) and \
                (x < poly[i][0] + (poly[j][0] - poly[i][0]) * (y - poly[i][1]) /
                                  (poly[j][1] - poly[i][1])):
            c = not c
        j = i
    return c

def pr(a):
    """ prune prior to convex hull

    `<http://mindthenerd.blogspot.com/2012/05/
    fastest-convex-hull-algorithm-ever.html>`_.

    A = (Ax, Ay) which maximizes x-y 
    B = (Bx, By) which maximizes x+y 
    C = (Cx, Cy) which minimizes x-y
    D = (Dx, Dy) which minimizes x+y

    """
    a_minus = np.nansum(a * [1, -1], axis=1)
    a_plus = np.nansum(a, axis=1)
    xy0, xy1 = np.nanmin(a_minus), np.nanmax(a_minus)
    xy2, xy3 = np.nanmin(a_plus), np.nanmax(a_plus)
    A = a[np.where(a_minus == xy0)[0]]  # min x-y
    B = a[np.where(a_minus == xy1)[0]]   # max x-y
    C = a[np.where(a_plus == xy2)[0]]   # min x+y
    D = a[np.where(a_plus == xy3)[0]]   # max x+y
    
    stack = np.vstack((A, B, C, D))
    x1 = max(stack[2][0], stack[3][0])  # max Cx, Dx
    x2 = min(stack[0][0], stack[1][0])  # min Ax, Bx
    y1 = max(stack[0][1], stack[3][1])  # max Ay, Dy
    y2 = min(stack[1][1], stack[2][1])  # min By, Cy
    # (x₂, y₁), (x₂, y₂), (x₁, y₂), (x₁, y₁)
    n_arr = np.array([[x2, y1], [x2, y2], [x1, y2], [x1, y1]]) 
    # Need pip this only provides the first run
    return A, B, C, D, x1, x2, y1, y2, n_arr

#z = np.random.randint(0, 10, size=(10,2))
#a = np.copy(z)
a = np.array([[6, 7], [1, 1], [0, 6], [6, 3], [3, 8],
               [4, 0], [9, 1],[7, 0], [0, 5], [2, 2]])

    
def prune(a):
    """Prune the points to reduce the point cloud.  Pruning is done from the
    centre of the data. The distance from the centre to each point is
    calculated and the 25, 50 and 75th percentiles are obtained.
    The indices of the distances > threshold (eg 50%) are used to pull out 
    a subsample of points greater than distance from the centre.
    Concave hull can be determined for the remaining points, as long as the
    the original point set doesn't perforate the exterior of the point cloud

    Example:
    -------
    >>> a = np.load('C:/Arc_projects/Ice/Data/fordan3_a.npy')
    >>> a.shape   # (50409, 2)
    >>> sub, low_median_up = prune(a)
    >>> low_median_up  # distance at 25, 50, 75 (314.12, 444.72, 548.69)
    >>> sub.shape  # (25201, 2)
    """
    def e_2d(p, a):
        """ array points to point distance... mini e_dist
        """
        p = np.asarray(p)
        diff = a - p[np.newaxis, :]
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))
    #
    from scipy.spatial import cKDTree  #, KDTree
    cent = a.mean(axis=0)
    t = cKDTree(a)
    xy_min = t.mins   # a.min(axis=0)  equivalent
    xy_max = t.maxes  # a.max(axis=0)
    dxdy = xy_max - xy_min
    delta = int(round(min(dxdy)))
    max_dist = delta - (delta % 100) + 100
    lens = [[i, len(t.query_ball_point(cent, i))]
            for i in range(0, max_dist, 100)]
    lens = np.asarray(lens)
    # or
    dists = e_2d(cent, a)
    dist_median = np.median(dists)
    low_median_up = np.percentile(dists, [25, 50, 75])  # quartiles and median
    w = np.where(dists > dist_median)[0]
    sub = a[w]
    return sub, low_median_up


def nd_2_struct(a):
    """convert ndarray to a structured array with columns

    Examples:
    ---------
    >>> shp = (3, 4)
    >>> a = np.arange(np.product(shp)).reshape(shp)
    >>> nd_2_columns(a)
    array([(0., 4.,  8.), (1., 5.,  9.), (2., 6., 10.), (3., 7., 11.)],
           dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
    >>> # --- 3D array ----
    >>> shp =(2,3,4)
    >>> a = np.arange(np.product(shp)).reshape(shp)
    >>> nd_2_columns(a)
    array([[( 0., 12.), ( 1., 13.), ( 2., 14.), ( 3., 15.)],
           [( 4., 16.), ( 5., 17.), ( 6., 18.), ( 7., 19.)],
           [( 8., 20.), ( 9., 21.), (10., 22.), (11., 23.)]],
          dtype=[('a', '<f8'), ('b', '<f8')])
    
    Reference:
    ----------
    `<https://stackoverflow.com/questions/54407313/use-structured-array-to-
    name-axis-in-numpy-array>`_.

    Extras:
    -------
    You can stack a 3D array as a column array like the structured version.

    >>> r =[a[i].ravel() for i, _ in enumerate(a)]
    >>> np.array(r)    # ---- for horizontal version or
    >>> np.array(r).T  # ---- for vertical version
    """
    names = "abcdefghijkl"
    shp = a.shape
    cols = shp[0] 
    dt = [(n, 'float') for _, n in enumerate(names[:cols])]
    z = np.zeros(shp[1:], dtype=dt)
    flds = z.dtype.names
    for i, sub in enumerate(a):
        name = flds[i]
        z[name] = sub
    return z

    

'''
de =e_dist(a,a)
t.query_ball_point(a, 20)
#de * np.where(de <= 20, 1, 0)  # for a query and to set things to 0
de[np.diag_indices_from(de)] = np.inf
np.fill_diagonal(de, np.nan)
np.argmin(de, axis=1)
#Out[95]: array([9, 6, 0, 7, 5, 4, 9, 3, 9, 8], dtype=int64)

idx = np.argsort(de, axis=1)

#array([[9, 6, 8, 1, 3, 2, 7, 4, 5, 0],
#       [6, 9, 0, 8, 3, 4, 7, 5, 2, 1],
#       [0, 7, 9, 3, 8, 6, 1, 5, 4, 2],
#       [7, 1, 0, 5, 6, 4, 9, 8, 2, 3],
#       [5, 1, 3, 6, 7, 9, 0, 8, 2, 4],
#       [4, 3, 7, 1, 6, 0, 9, 8, 2, 5],
#       [9, 1, 8, 0, 3, 4, 7, 2, 5, 6],
#       [3, 1, 0, 5, 6, 2, 9, 4, 8, 7],
#       [9, 6, 0, 1, 3, 2, 7, 4, 5, 8],
#       [8, 6, 0, 1, 3, 7, 2, 4, 5, 9]], dtype=int64)

dist = nn_kdtree(a, 1, True, False, True)
'''
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    print("Script path {}".format(script))

#a0 = 'C:/Arc_projects/Ice/Data/fordan3_a.npy'
#a1 = 'C:/Arc_projects/Ice/Data/fordan3_b.npy'
#a = np.load(a0)

import numpy as np
#import arcpy

def mean_split(a, minSize=3):
    """split at means"""
    def slice_array(a):
        m = np.mean(a)
        yes = a <= m  # check for a less than the overal mean
        a_left, a_rght = a[~yes], a[yes]  # slice the arrays
        return m, a_left, a_rght
    # ----
    m, L, R = slice_array(a)
    m0, L, _ = slice_array(L)
    m1, _, R = slice_array(R)
    means = [m0, m, m1]
    while (len(L) > minSize) and (len(R) > minSize):
        print("lefts", len(L), len(R))
        m0, L, _ = slice_array(L)
        m1, _, R = slice_array(R)
        means.extend([m0, m1])
#    while (len(R) > minSize):
#        print("Rights", len(L), len(R))
#        m0, l, _ = slice_array(L)
#        m1, _, R = slice_array(R)
#        means.extend([m0, m1])       
    return sorted(means)

#a = np.array([59, 43, 60, 97, 18,  3, 96, 41, 62, 60])
#a = np.arange(1, 20)
#
#tbl = r"C:\Arc_projects\Table_tools\Table_tools.gdb\pnts_2K_normal"
#in_fld = "Norm"
#out_fld = "New_class"
#
## ---- Do some work
## (1)  Get the field from the table and make it a simple array
#arr = arcpy.da.TableToNumPyArray(tbl, ['OID@', in_fld], "", True, None)
#a = arr[in_fld]
#
## (2)  Set up for the results
#out_arr = np.copy(arr)
#out_arr.dtype.names = ['OID@', out_fld]
#
## (3)  Run the mean_split script... note minSize!!! set it smartly or...
#means = mean_split(a, minSize=100)
#classed = np.digitize(a, bins=means)
#
## (4)  Send the results to the output array and add it back to arcgis pro
#out_arr[out_fld] = classed
#arcpy.da.ExtendTable(tbl, 'OID@', out_arr, 'OID@')


# -----
#a = np.arange(3*4*2).reshape(3,4,2)
#powr = np.array([2])  # or [2,2,2] if a.shape[0] == 3
## simulating power
#b = np.repeat(powr, a.shape[0])
#c = np.power(a, b[:, np.newaxis, np.newaxis])
#d = a ** b[:, None, None] # smoothest

# ----
#from scipy.spatial import Delaunay
#a0, a1, a = _common_(proj_pth, fn, pth, gdb, buff)
#arr, SR, new_names, shp_fld =a0
#all_pnts, cch_pnts, cch_ply, inside_pnts, cch_inside = a1
#xy = arr[['X', 'Y']]
#xy_v = _view_(xy)
#tri = Delaunay(xy_v)
#points = tri.points
#simplexes = tri.simplices
#alf = get_alpha_complex(0.2, points, simplexes)
#for s in alpha_complex: #fill in the triangles of the alpha complex
#    A=pts[s[0]]
#    B=pts[s[1]]
#    C=pts[s[2]]



# https://stackoverflow.com/questions/15722324/sliding-window-in-numpy
# cool
# indexer = np.arange(6)[None, :] + np.arange(4)[:, None]
# np.indices((2,2))

"""
preferred way to create a structured array... makes sense... note the shape

data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
print(data.dtype)

"""
"""
# example on geonet

np.random.RandomState(123)
a = np.random.randint(0, 5, size=(10,10))
uniq, cnts = np.unique(a, return_counts=True)
per = cnts/cnts.sum() *100.
dt = [('Value', '<i4'), ('Count', '<i4'), ('Percent', '<f8')]
out = np.zeros(uniq.size, dtype=dt)
out['Value'] = uniq
out['Count'] = cnts
out['Percent'] = per

"""
#xyz = np.load(r"c:\git_dan\arraytools\data\xyz.npy")
#a = _view_(xyz[['POINT_X', 'POINT_Y', 'POINT_Z']])
#len_3d = e_leng(a)
'''
def spiral_archim(N, n, clockwise=True, reverse=False):
    """Create an Archimedes spiral in the range 0 to N points with 'n' steps
    : between each incrementstep.  You could use np.linspace
    :Notes: When n is small relative to N, then you begin to form rectangular
    :  spirals, like rotated rectangles
    :Tried:  N = 1000, n = 30
    """
    rnge = np.arange(0.0, N+1.0)
    if clockwise:
        rnge = rnge[::-1]
    phi = rnge/n * np.pi
    xs = phi * np.cos(phi)
    ys = phi * np.sin(phi)
    if reverse:
        tmp = np.copy(xs)
        xs = ys
        ys = tmp
    xy = np.c_[xs, ys]
    wdth, hght = np.ptp(xy, axis=0)
    return xs, ys, xy 
'''
#arcpy.management.SelectLayerByAttribute("pnts_2000", "NEW_SELECTION", "Text01 = 'C'", None)
#import arcpy
#import time
#gdb = r"C:\Git_Dan\arraytools\array_tools_testing\array_tools.gdb"
#fc = gdb + r"\pnts_2000"
#lyr = r"C:\Arc_projects\Polygon_lineTools\pnts_2000_lyr.lyrx"
#qry = "Text01 = 'C'"
#uniq = ['A', 'Aa', 'Aaa', 'B', 'Bbbbb', 'C', 'Ccccc', 'D', 'Dddd', 'None']
#def test(lyr, uniq):
#    """ """
#    start = time.time()
#    qry = "Text01 = '{}'".format(uniq)
#    r = arcpy.management.SelectLayerByAttribute(lyr, "NEW_SELECTION", qry, None)
#    stop = time.time()
##    arcpy.management.SelectLayerByAttribute(lyr, "CLEAR_SELECTION") #, qry, None)
#    return (stop - start)

#gdb = r"C:\Git_Dan\arraytools\array_tools_testing\array_tools.gdb"
#fc = gdb + r"\pnts_2000"
#fld = 'Text01'
#flds = arcpy.ListFields(fc)
#fld_info = np.array([[f.name, f.type] for f in flds])
#txt_flds = fld_info[fld_info[:, 1] == 'String']
#num_flds = []
#for i, n in enumerate(fld_info[:, 1]):
#    if n in ('Integer', 'SmallInteger', 'Double', 'Single'):
#        num_flds.append(fld_info[i])
#num_flds = np.array(num_flds)
#
#
#def summary(fc, fld):
#    """ """
#    a = arcpy.da.TableToNumPyArray(fc, fld)
#    uniq = np.unique(a[fld])
#    result =  np.array([[u, a[a[fld] == u].size] for u in uniq])
#    out = np.zeros(len(result), dtype=[('Class', 'U8'), ('Count', '<i4')])
#    out['Class'] = result[:, 0]
#    out['Count'] = result[:, 1]
#    return out

#results = [[f, summary(fc, f)] for f in txt_flds[:,0]]
#for i in results:
#    print("Field ... {}".format(i[0]))
#    prn_struct(i[1])
   
# cool  # symmetric matrix
#k = np.arange(0, 11)[:, np.newaxis] + np.arange(0, -11, -1)

#abs(k)
#Out[3]: 
#array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
#       [ 1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
#       [ 2,  1,  0,  1,  2,  3,  4,  5,  6,  7,  8],
#       [ 3,  2,  1,  0,  1,  2,  3,  4,  5,  6,  7],
#       [ 4,  3,  2,  1,  0,  1,  2,  3,  4,  5,  6],
#       [ 5,  4,  3,  2,  1,  0,  1,  2,  3,  4,  5],
#       [ 6,  5,  4,  3,  2,  1,  0,  1,  2,  3,  4],
#       [ 7,  6,  5,  4,  3,  2,  1,  0,  1,  2,  3],
#       [ 8,  7,  6,  5,  4,  3,  2,  1,  0,  1,  2],
#       [ 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,  1],
#       [10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0]])

#import arcpy
#in_fc = r'C:\Arc_projects\profile_maker\profiler.gdb\transect'
#desc = arcpy.da.Describe(in_fc)
#SR = desc['spatialReference']
#out_fc = r'C:\Arc_projects\profile_maker\profiler.gdb\transect_split'
#shps = []



#line = arcpy.da.SearchCursor(in_fc, ("SHAPE@",)).next()[0]
#arcpy.CopyFeatures_management(
#        [line.segmentAlongLine(i/float(out_count),
#           ((i+1)/float(out_count)),  True) for i in range(0, out_count)], out_fc)
    """
