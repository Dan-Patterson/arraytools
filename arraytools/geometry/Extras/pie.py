# -*- coding: UTF-8 -*-
"""
:Script:   pie.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-02-21
:Purpose:  point in extent
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import arcpy
from textwrap import dedent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=5, linewidth=80, precision=2, suppress=True,
                    threshold=30, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def pie(pnts, ext):
    """Point in extent.
    :Performs a logical_and to find points within a box/ext
    :comp = np.logical_and( great-eq LB, less RT) # logic
    :inside = np.where(np.prod(comp, axis=1) == 1) # logic
    """
    L, B, R, T = ext.flatten()
    comp = np.logical_and(([L, B] <= pnts), (pnts <= [R, T]))
    idx = np.where(comp[:, 0] * comp[:, 1] == 1)[0]
    ids = np.arange(len(pnts))
    inside = pnts[idx]
    idx_o = np.delete(ids, idx)
    outside = pnts[idx_o]
    return inside,  outside


def _extent(ext):
    """construct the extent rectangle from the extent points"""
    box = [ext[0], [ext[0][0], ext[1][1]], ext[1],
           [ext[1][0], ext[0][1]], ext[0]]
    ext_rect = np.array(box)
    return ext_rect


def _plot_pnts(inside, outside, ext):
    """plot the points inside and outside the extent rectangle
    """
    import matplotlib.pyplot as plt
    ext_rect = _extent(ext)
    xmin = np.min(outside[:, 0])
    xmax = np.max(outside[:, 0])
    ymin = np.min(outside[:, 1])
    ymax = np.max(outside[:, 1])
    plt.figure(1)
    a1 = plt.subplot(111)
    a1.axis('equal')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.plot(inside[:, 0], inside[:, 1], "bo",
             outside[:, 0], outside[:, 1], "rs",
             ext_rect[:, 0], ext_rect[:, 1], "g-")
    plt.show()


def main(in_FC=None, N=1000):
    """Run with demo file"""
    if in_FC is None:
        in_FC = r"C:\Git_Dan\a_Data\testdata.gdb\Carp_5x5km"
        out_FC = r"C:\GIS\Pro_base\data\Pro_base.gdb\xy"
    SR = arcpy.Describe(in_FC).spatialReference
    aoi = arcpy.Describe(in_FC).extent
    flds = ('OID@', 'SHAPE@X', 'SHAPE@Y')
    arr = arcpy.da.FeatureClassToNumPyArray(in_FC, spatial_reference=SR,
                                            field_names=flds,
                                            explode_to_points=True)
    arr.dtype = [('ID', '<i4'), ('X', '<f8'), ('Y', '<f8')]
    # aoi = desc.extent
    ext = np.array([round(i)
                    for i in [aoi.XMin, aoi.YMin, aoi.XMax, aoi.YMax]])
    L, B, R, T = ext
    x = np.random.randint(L, R, N)
    y = np.random.randint(B, T, N)
    pnts = np.c_[x, y]
    ext = np.array([[L + 2000, B + 2000], [R - 2000, T - 2000]])
    inside, outside = pie(pnts, ext)
    inside = np.array(inside)
    out_pnts = np.zeros((len(pnts),), dtype=[('X', '<f8'), ('Y', '<f8')])
    out_pnts['X'] = pnts[:, 0]
    out_pnts['Y'] = pnts[:, 1]
    # ---- uncomment below to create a new output file ----
    #arcpy.da.NumPyArrayToFeatureClass(out_pnts, out_FC, ('X', 'Y'), SR)
    # ----
    frmt = """
    :Point in extent... 'pie' ...
    :Input points N=({})
    :min X,Y ... {}
    :max X,Y ... {}
    :Extents....
    {!r:}
    :Result..... inside ({})  outside ({})
    :inside.....
    {!r:}
    """
    args = [len(pnts), np.min(pnts, axis=0), np.max(pnts, axis=0),
            ext, len(inside), len(outside), inside]
    print(dedent(frmt).format(*args))
    return arr, pnts, inside, outside


def _demo(N=1000, plot=False):
    """
    : -
    """
    import timeit
    rep = 5  # repeats for the below
    # pnts = np.random.random_sample((N, 2)) * 10
    L, B, R, T = np.array([340000., 5020000, 345000, 5025000])
    x = np.random.randint(L, R, N) * 1.0
    y = np.random.randint(B, T, N) * 1.0
    pnts = np.c_[x, y]
    ext = np.array([[L + 2000, B + 2000], [R - 2000, T - 2000]])
    inside, outside = pie(pnts, ext)
    inside = np.array(inside)
    frmt = """
    :Point in extent... 'pie' ...
    :Points  {}
    :min ... {}
    :max ... {}
    :Extents...
    {}
    :Result....
    {}
    """
    args = [len(pnts), np.min(pnts, axis=0), np.max(pnts, axis=0),
            ext, len(inside)]
    print(dedent(frmt).format(*args))
    set_up = "from __main__ import pie"
    t = timeit.timeit(stmt='pie', setup=set_up, number=rep)
    print("Time results: {} s, for {} repeats".format(t, rep))
    if plot:
        _plot_pnts(inside, outside, ext)
    return pnts, inside, outside


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Make some points for testing, create and extent,
    :time each option and optionally graph
    Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    pnts, inside, outside = _demo(N=1000, plot=True)
    arr, pnts, inside, outside = main(in_FC=None, N=1000)
