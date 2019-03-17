# -*- coding: utf-8 -*-
"""
=======
fc_demo
=======

Script : fc_demo.py

Author : Dan_Patterson@carleton.ca

Modified : 2019-03-14

Purpose :  

Notes:

References:

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import numpy as np
import arcpy

from fc import  _xy_idx
from geom_properties import e_leng

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=140, precision=2, suppress=True,
                    threshold=180, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

#script = sys.argv[0]  # print this should you need to locate the script

# ===========================================================================
# ---- def section: def code blocks go here ---------------------------------


# ===========================================================================
# ---- main section: testing or tool run ------------------------------------
#
def _common_():
    """Stuff common to _demo_ and _tool()
    """
    script = sys.argv[0]
    return script


def _demo_():
    """Run in spyder
    """
    in_fc0 = r'C:\Arc_projects\profile_maker\profiler.gdb\rotated'
    desc = arcpy.da.Describe(in_fc0)
    SR = desc['spatialReference']
#    in_fc1 = r'C:\Arc_projects\profile_maker\profiler.gdb\transect_split'
    script = _common_()
    msg0 = "\nRunning... {} in Spyder\n".format(script)
    a, idx0 = _xy_idx(in_fc0)
#    s, idx1 = _xy_idx(in_fc1)
#    with arcpy.da.SearchCursor(in_fc, 'SHAPE@') as cursor:
#        for row in cursor:
#            shps.append(row[0])
    # ---- now this could be iterated for multiple input polylines
#    shp = shps[0]
#    shp_len = shp.length
#    step = 100.
#    divs = np.arange(0, shp_len + step, step)
#    frum = divs[0: -1]
#    too = divs[1:]
#    polys = [shp.segmentAlongLine(frum[i], too[i], False) for i in range(len(frum))]
    
#    pnts = [(p.firstPoint.X, p.firstPoint.Y) for p in polys]    # get the segment
#    pnts.extend((polys[-1].lastPoint.X, polys[-1].lastPoint.Y)) # start/end points  
    #arcpy.CopyFeatures_management(polys, out_fc)

    return a, idx0  #, s, idx1

def _tool_():
    """run from a tool in arctoolbox in arcgis pro
    """
    script = _common_()
    msg0 = "\nRunning... {} in in ArcGIS Pro\n".format(script)
    return msg0

def line_split(poly, dist):
    """Split a line at finite distances
    """
    a_0 = poly[0]  # first point
    a_s = a - a_0
    dxdy = a_s[1:, :] - a_s[:-1, :]                       # coordinate differences
    leng = np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))
    c_sum = np.cumsum(leng)              # cumulative length
    split_at = np.arange(dist, tot + dist, dist)  # splitters to subdivide lengths
    sa = np.digitize(c_sum, split_at)    # a temporary value 
    frum = np.where(np.diff(sa) == 1)[0] + 1  # actual indices to split on
    too = frum + 1
    subs = np.split(a_s, too)    # split array
    return subs

def func(a, dist):
    """working with stuff
    
    n = ('xs', 'ys', 'dx', 'dy', 'leng', 'cumleng', 'steps',
         'deltaX', 'deltaY')
    k = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8','f8','f8')
    dt = {'names': n, 'formats': k}
    z0 = z.view(dtype=dt).squeeze()
    from arraytools.frmts import prn
    prn(z0)
    """
    z = np.zeros((a.shape[0],9))
    z[:, :2] = a                                 # xys : x, y coordinates
    z[1:, 2:4] = dxdy = z[1:, :2] - z[:-1, :2]   # sequential differences
    z[1:, 4] = np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))
    z[:, 5] = np.cumsum(z[:,4])                  # cumulative distance
    z[1:, 6] = steps = z[1:, 4] / dist           # steps to create
    z[1:, 7:] = deltas = dxdy/(steps.reshape(-1, 1))

    N = len(a) - 1  # number of segments
    pnts = np.empty((N,), dtype='O')
    for i in range(N):              # cycle through the segments and make
        num = np.arange(steps[i])   # the new points
        pnts[i] = np.array((num, num)).T * deltas[i] + a[i]
    a0 = a[-1].reshape(1,-1)
    final = np.concatenate((*pnts, a0), axis=0)
    return z, pnts, final

# ===========================================================================
# ---- main section: testing or tool run ------------------------------------
#
#if len(sys.argv) == 1:
##    a, idx0 = _demo_()
#    a = np.array([[ 0, 0], [ 0, 100], [100, 100], [100,  80],
#                  [ 20,  80], [ 20, 20], [100, 20], [100, 0], [ 0, 0]])
#    a = a/10
##    a = np.array([[ 0, 0], [ 400, 300], [400, 0], [0, 0]])
#    dist = 4
##    uni = line_split(a, dist)  # split the lines
#    z, pnts, final = func(a, dist)
#else:
#    msg = _tool_()



# ==== Processing finished ====
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
#    msg = _demo_()
# a = np.array([[ 0, 0], [ 0, 100], [100, 100], [100,  80],
#                      [ 20,  80], [ 20, 20], [100, 20], [100, 0], [ 0, 0]])
xy = np.array([[0, 0], [1, 0], [1, 1], [2, 3], [3, 4],
               [3, 5], [3, 6], [6, 7], [7, 8], [9, 9]])
x = xy[:, 0]
y = xy[:, 1]
xy_T = np.einsum('ij->ji', xy)    # transpose of the array
xy_sum = np.einsum('ij->', xy)    # 2D sum
row_sum = np.einsum('ij->i', xy)
col_sum = np.einsum('ij->j', xy)
xsq = np.einsum('ij->i', xy)
ysq = np.einsum('ij->j', xy)
# double elements
xsq_ysq = np.einsum('ij, ij->ij', xy, xy)  # square element-wise
row_sum2 = np.einsum('ij, ij->i', xy, xy)
col_sum2 = np.einsum('ij, ij->j', xy, xy)
way_cool = np.einsum('ij, jk->ik', xy, xy.T)
not_sure = np.einsum('ij, jk->ik', xy.T, xy)
ns0 = np.einsum('ij, jk->jk', xy, xy.T)
ns1 = np.einsum('ij, jk->ij', xy, xy.T)  # matrix multiplication?
#
avg = np.mean(xy, axis=0)
xy_avg = np.einsum('ij->ij', (xy-avg))
xy_avg_sq = np.einsum('ij,ij->i', (xy-avg), (xy-avg))  # by row
xy_avg_sq_colsum =np.einsum('ij,ij->j', (xy-avg), (xy-avg))


import numpy as np 
import matplotlib.pyplot as plt 
  
def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x)  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y)   
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x   
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x  
    return(b_0, b_1) 

def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30)   
    # predicted response vector 
    y_pred = b[0] + b[1]*x  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g")   
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y')   
    # function to show plot 
    plt.show() 
    
def ein_corr(x, y):
    """einsum correlation etc
    """
    n = x.size
    x_m = x.mean()   # np.add.reduce(x) / n  = np.einsum('i->', x) / n
    y_m = y.mean()   # np.add.reduce(y) / n  = np.einsum('i->', y) / n
    x_std = x.std()
    y_std = y.std()
    xx_sum = (x*x).sum()  # np.einsum('i,i->', x, x)
    xy_sum = (x*y).sum()  # np.einsum('i,i->', x, y)
    ss_xx = xx_sum - x_m * x_m * n
    ss_xy = xy_sum - x_m * y_m * n
    b = ss_xy / ss_xx 
    a = y_m - b*x_m
    # a + bx = y
    sxy = ((x - x_m)*(y - y_m)).mean() # np.einsum('i, i->', x-x_m, y-x_m)/n
    corr_coeff = sxy / (x_std*y_std)    # np.corrcoef(x, y)[0, 1]
    return (a, b, corr_coeff)

a = np.arange(4).reshape(4)
b = np.arange(4*2).reshape(4, 2)
c = np.arange(3*4*2).reshape(3, 4, 2)
tests = [np.sum(a),
np.sum(a, axis=0),
np.sum(a),
np.sum(b),
np.sum(b, axis=0),
np.sum(b, axis=1),
np.sum(b, axis=(0,1)),
np.sum(c),
np.sum(c, axis=(0,1,2)),
np.sum(c, axis=(0,1)),
np.sum(c, axis=(0,2)),
np.sum(c, axis=(1,2)),
np.sum(c, axis=1),
np.sum(c, axis=2),
np.sum(c, axis=0)]

#arg [a, a, 0]
"np.sum({})\n".format(a)
