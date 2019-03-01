# -*- coding: UTF-8 -*-
"""
========
vincenty
========

Script : vincenty.py

Author : Dan.Patterson@carleton.ca

Created : 2014-10

Modified : 2019-02-27

Purpose :
    Calculates the Vincenty Inverse distance solution for 2 long/lat pairs

References
----------
`<http://www.movable-type.co.uk/scripts/latlong-vincenty.html>`_.  java code

T Vincenty, 1975 "Direct and Inverse Solutions of Geodesics on the
Ellipsoid with application of nested equations", Survey Review,
vol XXIII, no 176, 1975

`<http://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf>`_.

`<https://github.com/geopy/geopy/blob/master/geopy/distance.py>`_.

Notes
-----
>>> atan2(y, x) # or atan2(sin, cos) not like Excel

used fmod(x, y) to get the modulous as per python

`<http://stackoverflow.com/questions/34552284/vectorize-haversine-distance-
computation-along-path-given-by-list-of-coordinates>`_.

Returns
-------
Distance in meters, initial and final bearings (as an azimuth from N)

Examples::

    data = np.array([[-75.0, 45.0, -75.0, 46.0],
                     [-75.0, 46.0, -75.0, 45.0],
                     [-76.0, 45.0, -75.0, 45.0],
                     [-75.0, 45.0, -76.0, 45.0],
                     [-76.0, 46.0, -75.0, 45.0],
                     [-75.0, 46.0, -76.0, 45.0],
                     [-76.0, 45.0, -75.0, 46.0],
                     [-75.0, 45.0, -76.0, 46.0],
                     [-90.0,  0.0,   0.0,  0.0],
                     [-75.0,  0.0, -75.0, 90.0]])
    orig = data[:, :2]
    dest = data[:, -2:]

: long0  lat0  long1  lat1   dist       initial    final  head to
: -75.0, 45.0, -75.0, 46.0   111141.548   0.000,   0.000   N
: -75.0, 46.0, -75.0, 45.0   111141.548 180.000, 180.000   S
: -76.0, 45.0, -75.0, 45.0    78846.334  89.646,  90.353   E
: -75.0, 45.0, -76.0, 45.0    78846.334 270.353, 269.646   W
: -76.0, 46.0, -75.0, 45.0   135869.091 144.526, 145.239   SE
: -75.0, 46.0, -76.0, 45.0   135869.091 215.473, 214.760   SW
: -76.0, 45.0, -75.0, 46.0   135869.091  34.760,  35.473   NE
: -75.0, 45.0, -76.0, 46.0   135869.091 325.239, 324.526   NW
: -90.0,  0.0    0.0   0.0 10018754.171  90.000   90.000   1/4 equator
: -75.0   0.0  -75.0  90.0 10001965.729   0.000    0.000   to N pole
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----

import sys
import numpy as np
import math
from textwrap import dedent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

__all__ = ['vincenty_cal',
           'vincenty']

# ---- functions ----

def vincenty_cal(long0, lat0, long1, lat1):
    """return the distance on the ellipsoid between two points using
    Vincenty's Inverse method

    Notes
    -----
    `a`, `b` : numbers
        semi-major and minor axes WGS84 model
    `f` : number
        inverse flattening
    `L`, `dL` : number
        delta longitude, initial and subsequent
    `u0`, `u1` : number
        reduced latitude
    `s_sig` : number
        sine sigma
    `c_sig` : number
        cosine sigma
    """
    a = 6378137.0
    b = 6356752.314245
    ab_b = (a**2 - b**2)/b**2
    f = 1.0/298.257223563
    twoPI = 2*math.pi
    dL = L = math.radians(long1 - long0)
    u0 = math.atan((1 - f) * math.tan(math.radians(lat0)))  # reduced latitudes
    u1 = math.atan((1 - f) * math.tan(math.radians(lat1)))
    s_u0 = math.sin(u0)
    c_u0 = math.cos(u0)
    s_u1 = math.sin(u1)
    c_u1 = math.cos(u1)
    # ---- combine repetitive terms ----
    sc_01 = s_u0*c_u1
    cs_01 = c_u0*s_u1
    cc_01 = c_u0*c_u1
    ss_01 = s_u0*s_u1
    #
    lambdaP = float()
    max_iter = 20
    # first approximation
    cnt = 0
    while (cnt < max_iter):
        s_dL = math.sin(dL)
        c_dL = math.cos(dL)
        s_sig = math.sqrt((c_u1*s_dL)**2 + (cs_01 - sc_01*c_dL)**2)  # eq14
        if (s_sig == 0):
            return 0
        c_sig = ss_01 + cc_01*c_dL                      # eq 15
        sigma = math.atan2(s_sig, c_sig)                # eq 16
        s_alpha = cc_01*s_dL/s_sig                      # eq 17
        c_alpha2 = 1.0 - s_alpha**2
        if c_alpha2 != 0.0:
            c_sigM2 = c_sig - 2.0*s_u0*s_u1/c_alpha2    # eq 18
        else:
            c_sigM2 = c_sig
        C = f/16.0 * c_alpha2*(4 + f*(4 - 3*c_alpha2))  # eq 10
        lambdaP = dL
        # dL => equation 11
        dL = L + (1 - C)*f*s_alpha*(sigma +
                                    C*s_sig*(c_sigM2 +
                                             C*c_sig*(-1.0 + 2*c_sigM2**2)))
        #
        if (cnt == max_iter):          # is it time to bail?
            return 0.0
        elif((math.fabs(dL - lambdaP) > 1.0e-12) and (cnt < max_iter)):
            cnt += 1
        else:
            break
    # ---- end of while ----
    uSq = c_alpha2 * ab_b
    A = 1 + uSq/16384.0 * (4096 + uSq*(-768 + uSq*(320 - 175*uSq)))  # eq 3
    B = uSq/1024.0 * (256 +  uSq*(-128 + uSq*(74 - 47*uSq)))         # eq 4
    d_sigma = B*s_sig*(c_sigM2 +
                      (B/4.0)*(c_sig*(-1 + 2*c_sigM2**2) -
                      (B/6.0)*c_sigM2*(-3 + 4*s_sig**2)*(-3 +
                      4*c_sigM2**2)))
    # d_sigma => eq 6
    dist = b*A*(sigma - d_sigma)                                     # eq 19
    alpha1 = math.atan2(c_u1*s_dL,  cs_01 - sc_01*c_dL)
    alpha2 = math.atan2(c_u0*s_dL, -sc_01 + cs_01*c_dL)
    # normalize to 0...360  degrees
    alpha1 = math.degrees(math.fmod((alpha1 + twoPI), twoPI))        # eq 20
    alpha2 = math.degrees(math.fmod((alpha2 + twoPI), twoPI))        # eq 21
    return dist, alpha1, alpha2


def vincenty(data, as_array=True):
    """Calculate vincenty distance and bearings for single or multiple
    orig-dest pairs of longitude/latitude coordinates

    Parameters
    ----------
    data : array-like
        long0, lat0, long1, lat1 as either a single entry or a Nx4 array
    as_array : boolean
        True, returns a structured array. False, returns an ndarray.
    """
    result = []
    for i, coords in enumerate(data):
        long0, lat0, long1, lat1 = data[i]
        row = vincenty_cal(long0, lat0, long1, lat1)
        row = np.concatenate((coords, row))
        result.append(row)
    if as_array:
        result = np.array(result)
        dt = [('Orig_long', '<f8'), ('Orig_lat', '<f8'), ('Dest_long', '<f8'),
              ('Dest_lat', '<f8'), ('Distance_m', '<f8'),
              ('Orig_Bearing', '<f8'), ('Dest_Bearing', '<f8')]
        return result.view(dt).squeeze()
    return np.array(result)

# ---- demos -----------------------------------------------------------------
#
def _demo_data():
    """some demo data"""
    d = [[-75.0, 45.0, -75.0, 46.0], [-74.0, 46.0, -74.0, 45.0],
         [-74.0, 45.0, -75.0, 45.0], [-75.0, 45.0, -76.0, 45.0],
         [-76.0, 46.0, -75.0, 45.0], [-75.0, 46.0, -76.0, 45.0],
         [-76.0, 45.0, -75.0, 46.0], [-75.0, 45.0, -76.0, 46.0],
         [-90.0,  0.0,   0.0,  0.0], [-75.0,  0.0, -75.0, 90.0]]
    return np.array(d)

def _demo_single():
    """ testing function edit as appropriate """
    coord = [-76.0, 45.0, -75.0, 46.0]  # SE

    a0, a1, a2, a3 = coord
    b0, b1, b2 = vincenty_cal(a0, a1, a2, a3)
    frmt = """
    :--------------------------------------------------------:
    :Vincenty inverse...
    :Longitude, Latitude
    :From: ({:>12.8f}, {:>12.8f})
    :To:   ({:>12.8f}, {:>12.8f})
    :Distance: {:>10.3f} m
    :Bearings...
    :  Initial {:>8.2f} deg
    :  Final   {:>8.2f} deg
    :--------------------------------------------------------:
    """
    print (dedent(frmt).format(a0, a1, a2, a3, b0, b1, b2))

def _demo_batch():
    """testing batch function"""

    frum_too = _demo_data()
    result = []
    for i, coords in enumerate(frum_too):
        long0, lat0, long1, lat1 = coords
        row = vincenty_cal(long0, lat0, long1, lat1)
        row = np.concatenate((coords, row[:-1]))
        result.append(row[:-1])
    return np.array(result)


#def _demo_arc():
#    """arcgis pro demo .... uncomment if you want to test
#    """
#    import arcpy
#    SR = arcpy.SpatialReference(4326)
#    frum_too = _demo_data()
#    frum = frum_too[:, :2].tolist()
#    too = frum_too[:, -2:].tolist()
#    pg_f = [arcpy.PointGeometry(arcpy.Point(i[0], i[1]), SR) for i in frum]
#    pg_t = [arcpy.PointGeometry(arcpy.Point(i[0], i[1]), SR) for i in too]
#    ang_dist = []
#    for i, f in enumerate(pg_f):
#        t = pg_t[i]
#        ft = f.angleAndDistanceTo(t)
#        ang_dist.append(tuple(frum[i] + too[i]) + ft)  # from-to 
#    return ang_dist
    
# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
#    #print("Script... {}".format(script))
#     ----- uncomment one of the  below  -------------------
#    _demo_single()
#    _demo_batch()
