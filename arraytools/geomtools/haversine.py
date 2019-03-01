# -*- coding: UTF-8 -*-
"""
=========
haversine
=========

Script : haversine.py

Author :  Dan.Patterson@carleton.ca

Created :  2014-10

Modified : 2019-02-27

Purpose :  Calculates the Haversine solution for long/lat pairs

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

>>> atan2(y,x) # or atan2(sin, cos) not like Excel

used fmod(x,y) to get the modulous as per python

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

>>> orig = data[:, :2]
>>> dest = data[:, -2:]
>>> from arraytools.frmts import prn
>>> result = haversine(orig, dest, as_array=True)
>>> prn(result)

Result::

    Array info...
    shape... (10,)  ndim... 1
    dtype... [('Orig_long', '<f8'), ('Orig_lat', '<f8'),
              ('Dest_long', '<f8'), ('Dest_lat', '<f8'),
              ('Distance_m', '<f8'), ('Bearing', '<f8')] 
    #
     id  Orig_long   Orig_lat   Dest_long   Dest_lat   Distance_m    Bearing  
    --------------------------------------------------------------------------
     000      -75.00      45.00      -75.00      46.00     111195.08      0.00
     001      -75.00      46.00      -75.00      45.00     111195.08    180.00
     002      -76.00      45.00      -75.00      45.00      78626.30     89.65
     003      -75.00      45.00      -76.00      45.00      78626.30    270.35
     004      -76.00      46.00      -75.00      45.00     135786.28    144.62
     005      -75.00      46.00      -76.00      45.00     135786.28    215.38
     006      -76.00      45.00      -75.00      46.00     135786.28     34.67
     007      -75.00      45.00      -76.00      46.00     135786.28    325.33
     008      -90.00       0.00        0.00       0.00   10007557.22     90.00
     009      -75.00       0.00      -75.00      90.00   10007557.22      0.00
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

# ---- functions ----

def haver_cal(orig, dest, as_array=True):
    """Haversine equation.

    Parameters
    ----------
    orig, dest : arrays
        Arrays of long/lat pairs.  Note suited for single orig-dest analysis
    as_array : boolean
        True, returns an array with a structured dtype. 
        False, returns the distance and bearings only
    Returns
    -------
    Distance and bearing

    Notes
    -----
    For sequential data like N-2

    >>> pairs = np.array([[-75.,  45.], [-75,  46.], [-74.,  46.],
    ...                   [-74.,  45.], [-75, 45]])
    >>> orig = pairs[:-1]
    >>> dest = pairs[1:]

    Bearing Formula

    >>> dy = sin(Δλ) * cos(φ2)
    >>> dx = cos(φ1) * sin(φ2) − sin(φ1) * cos(φ2) * cos(Δλ)
    >>> θ = arctan2(dy, dx)
    >>> φ1, λ1 is the start point,
    >>> φ2, λ2 the end point (Δλ is the difference in longitude)
    """
    o = np.deg2rad(orig)  # longitude, latitude
    d = np.deg2rad(dest)
    o_lon = o[:, 0]
    o_lat = o[:, 1]
    d_lon = d[:, 0]
    d_lat = d[:, 1]
    diff_lat = d_lat - o_lat
    diff_lon = d_lon - o_lon
    #
    # ---- distance calculations
    a0 = np.sin(diff_lat/2.)**2
    a1 = np.cos(o_lat) * np.cos(d_lat)
    a2 = np.sin(diff_lon/2.)**2
    dist = a0 + a1 * a2
    dist = 2.0 * 6371008.8 * np.arcsin(np.sqrt(dist))
    #
    # ---- bearing calculations
    dy = np.sin(diff_lon) * np.cos(d_lat)
    b0 = np.cos(o_lat) * np.sin(d_lat)
    dx = b0 - np.sin(o_lat) * np.cos(d_lat) * np.cos(diff_lon)
    bearing = np.degrees(np.arctan2(dy, dx))
    bearing = (bearing + 360) % 360
    if as_array:
        result = np.c_[(orig, dest, dist, bearing)]
        dt = [('Orig_long', '<f8'), ('Orig_lat', '<f8'), ('Dest_long', '<f8'),
              ('Dest_lat', '<f8'), ('Distance_m', '<f8'), ('Bearing', '<f8')]
        return result.view(dt).squeeze()
    return dist, bearing



def _demo():
    """ testing function edit as appropriate """
    pairs = np.array([[-75.0, 45.0, -75.0, 46.0],
                      [-75.0, 46.0, -75.0, 45.0],
                      [-76.0, 45.0, -75.0, 45.0],
                      [-75.0, 45.0, -76.0, 45.0],
                      [-76.0, 46.0, -75.0, 45.0],
                      [-75.0, 46.0, -76.0, 45.0],
                      [-76.0, 45.0, -75.0, 46.0],
                      [-75.0, 45.0, -76.0, 46.0],
                      [-90.0,  0.0,   0.0,  0.0],
                      [-75.0,  0.0, -75.0, 90.0]])
    orig = pairs[:, :2]
    dest = pairs[:, -2:]
    dist, bearing = haver_cal(orig, dest, as_array=False)
    result = np.c_[(orig, dest, dist, bearing)]
    hdr = """
    :--------------------------------------------------------:
    :Haverine inverse...
    :origins  destins
    :
    """
    hdr = ['Orig_long', 'Orig_lat', 'Dest_long', 'Dest_lat',
           'distance',  'bearing']
    frmt = "{:>10.2f} "*6
    print(("{:>10} "*6).format(*hdr))
    for i in result:
        print(frmt.format(*i))
#    print (dedent(frmt).format(orig, dest, dist, bearing))

# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
#    #print("Script... {}".format(script))
#     ----- uncomment one of the  below  -------------------
#    long0 = vin()  # not ready
#    demo()
