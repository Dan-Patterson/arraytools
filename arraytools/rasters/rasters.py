# -*- coding: UTF-8 -*-
"""
:Script:   rasters.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-12-16
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
from textwrap import dedent, indent
import numpy as np
from arraytools.tools import stride

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def expand_zone(a, zone=None, win=2):
    """Expand a value (zone) in a 2D array, normally assumed to represent a
    :  raster surface.
    :zone - the value/class to expand into the surrounding cells
    :win - select a (2, 2) or (3, 3) moving window
    :
    """
    msg = "\nYou need a zone that is within the range of values."
    if (zone is None):
        print(msg)
        return a, None
    elif (zone < a.min()) or (zone > a.max()):
        print(msg)
        return a, None
    if win not in (2, 3):
        win = 2
    p = [1, 0][win == 2]  # check for 2 or 3 in win
    ap = np.pad(a, pad_width=(1, p), mode="constant", constant_values=(0, 0))
    n, m = ap.shape
    if win == 2:
        a_c = ap[1:, 1:]  # for 2x2 even
    elif win == 3:
        a_c = ap[1:-1, 1:-1]  # for 3x3 odd
    a_s = stride(ap, win=(win, win), stepby=(win, win))  # stride the array
    r, c = a_s.shape[:2]
    out = []
    x = a_s.shape[0]
    y = a_s.shape[1]
    for i in range(x):
        for j in range(y):
            if zone in a_s[i, j]:
                out.append(1)
            else:
                out.append(0)
    out1 = np.asarray(out).reshape(r, c)
    out = np.repeat(np.repeat(out1, 2, axis=1), 2, axis=0)
    dx, dy = np.array(out.shape) - np.array(a.shape)
    if dx != 0:
        out = out[:dx, :dy]
    final = np.where(out == 1, zone, a_c)
    return final


def fill_arr(a, win=(3, 3)):
    """try filling an array"""
    fd = np.array([[32, 64, 128], [16, 0, 1], [8, 4, 2]])  # flow direction
#    if (zone < a.min()) or (zone > a.max()) or (zone is None):
#        print("\nYou need a zone that is within the range of values.")
#        return a, None
    if win[0] == 3:
        pr = 1
    else:
        pr = 0
    ap = np.pad(a, pad_width=(1, pr), mode="constant", constant_values=(0, 0))
    if win == (2, 2):
        a_c = ap[1:, 1:]  # for 2x2 even
        w, h = win
    elif win == (3, 3):
        w, h = win
        a_c = ap[1:-1, 1:-1]   # for 3x3 odd
    a_s = stride(a_c, win=win)  # stride the array
    r, c = a_s.shape[:2]
    out = []
    x = a_s.shape[0]
    y = a_s.shape[1]
    for i in range(x):
        for j in range(y):
            # do stuff
            sub = a_s[i, j].ravel()
            edges = np.asarray([sub[:4], sub[5:]]).ravel()
            e_min = edges[np.argmin(edges)]
            if sub[4] < e_min:
                out.append(e_min)
            else:
                out.append(sub[4])
    out = np.asarray(out).reshape(r, c)
    return out  # , a_s, ap, a_c


# (xx) reclass_vals .... code section
def reclass_vals(a, old_vals=[], new_vals=[], mask=False, mask_val=None):
    """Reclass an array of integer or floating point values.
    :Requires:
    :--------
    : old_vals - list/array of values to reclassify
    : new_bins - new class values for old value
    : mask - whether the raster contains nodata values or values to
    :        be masked with mask_val
    : Array dimensions will be squeezed.
    :Example:
    :-------
    :  array([[ 0,  1,  2,  3,  4],   array([[1, 1, 1, 1, 1],
    :         [ 5,  6,  7,  8,  9],          [2, 2, 2, 2, 2],
    :         [10, 11, 12, 13, 14]])         [3, 3, 3, 3, 3]])
    """
    a_rc = np.copy(a)
    args = [old_vals, new_vals]
    msg = "\nError....\nLengths of old and new classes not equal \n{}\n{}\n"
    if len(old_vals) != len(new_vals):
        print(msg.format(*args))
        return a
    old_new = np.array(list(zip(old_vals, new_vals)), dtype='int32')
    for pair in old_new:
        q = (a == pair[0])
        a_rc[q] = pair[1]
    return a_rc


# ----------------------------------------------------------------------
# (15) reclass .... code section
def reclass_ranges(a, bins=[], new_bins=[], mask=False, mask_val=None):
    """Reclass an array of integer or floating point values based on old and
    :  new range values
    :Requires:
    :--------
    : bins - sequential list/array of the lower limits of each class
    :        include one value higher to cover the upper range.
    : new_bins - new class values for each bin
    : mask - whether the raster contains nodata values or values to
    :        be masked with mask_val
    : Array dimensions will be squeezed.
    :Example:
    :-------
    :  z = np.arange(3*5).reshape(3,5)
    :  bins = [0, 5, 10, 15]
    :  new_bins = [1, 2, 3, 4]
    :  z_recl = reclass(z, bins, new_bins, mask=False, mask_val=None)
    :  ==> .... z                     ==> .... z_recl
    :  array([[ 0,  1,  2,  3,  4],   array([[1, 1, 1, 1, 1],
    :         [ 5,  6,  7,  8,  9],          [2, 2, 2, 2, 2],
    :         [10, 11, 12, 13, 14]])         [3, 3, 3, 3, 3]])
    """
    a_rc = np.zeros_like(a)
    if (len(bins) < 2):  # or (len(new_bins <2)):
        print("Bins = {} new = {} won't work".format(bins, new_bins))
        return a
    if len(new_bins) < 2:
        new_bins = np.arange(1, len(bins)+2)
    new_classes = list(zip(bins[:-1], bins[1:], new_bins))
    for rc in new_classes:
        q1 = (a >= rc[0])
        q2 = (a < rc[1])
        a_rc = a_rc + np.where(q1 & q2, rc[2], 0)
    return a_rc


# (16) scale .... code section
def scale_up(a, x=2, y=2, num_z=None):
    """Scale the input array repeating the array values up by the
    :  x and y factors.
    :Requires:
    :--------
    : a - an ndarray, 1D arrays will be upcast to 2D
    : x, y - factors to scale the array in x (col) and y (row)
    :      - scale factors must be greater than 2
    : num_z - for 3D, produces the 3rd dimension, ie. if num_z = 3 with the
    :    defaults, you will get an array with shape=(3, 6, 6)
    : how - if num_z != None or 0, then the options are
    :    'repeat', 'random'.  With 'repeat' the extras are kept the same
    :     and you can add random values to particular slices of the 3rd
    :     dimension, or multiply them etc etc.
    :Returns:
    :-------
    : a = np.array([[0, 1, 2], [3, 4, 5]]
    : b = scale(a, x=2, y=2)
    :   =  array([[0, 0, 1, 1, 2, 2],
    :             [0, 0, 1, 1, 2, 2],
    :             [3, 3, 4, 4, 5, 5],
    :             [3, 3, 4, 4, 5, 5]])
    :Notes:
    :-----
    :  a=np.arange(2*2).reshape(2,2)
    :  a = array([[0, 1],
    :             [2, 3]])
    :  f_(scale(a, x=2, y=2, num_z=2))
    :  Array... shape (3, 4, 4), ndim 3, not masked
    :   0, 0, 1, 1    0, 0, 1, 1    0, 0, 1, 1
    :   0, 0, 1, 1    0, 0, 1, 1    0, 0, 1, 1
    :   2, 2, 3, 3    2, 2, 3, 3    2, 2, 3, 3
    :   2, 2, 3, 3    2, 2, 3, 3    2, 2, 3, 3
    :   sub (0)       sub (1)       sub (2)
    :--------
    """
    if (x < 1) or (y < 1):
        print("x or y scale < 1... read the docs".format(scale_up.__doc__))
        return None
    a = np.atleast_2d(a)
    z0 = np.tile(a.repeat(x), y)  # repeat for x, then tile
    z1 = np.hsplit(z0, y)         # split into y parts horizontally
    z2 = np.vstack(z1)            # stack them vertically
    if a.shape[0] > 1:            # if there are more, repeat
        z3 = np.hsplit(z2, a.shape[0])
        z3 = np.vstack(z3)
    else:
        z3 = np.vstack(z2)
    if num_z not in (0, None):
        d = [z3]
        for i in range(num_z):
            d.append(z3)
        z3 = np.dstack(d)
        z3 = np.rollaxis(z3, 2, 0)
    return z3


def _demo():
    """
    : -
    """
    a = np.array([[9, 8, 2, 3, 4, 3, 5, 5, 2, 2],
                  [4, 1, 4, 2, 4, 2, 4, 2, 3, 2],
                  [5, 3, 5, 4, 5, 4, 5, 3, 1, 2],
                  [5, 2, 3, 1, 4, 4, 3, 5, 4, 3],
                  [2, 3, 2, 5, 5, 2, 5, 5, 4, 4],
                  [5, 3, 4, 4, 2, 1, 3, 2, 4, 3],
                  [3, 2, 3, 3, 3, 4, 3, 2, 4, 3],
                  [4, 5, 2, 3, 2, 2, 3, 1, 4, 4],
                  [3, 5, 5, 5, 2, 2, 4, 3, 4, 4],
                  [4, 5, 4, 5, 3, 2, 4, 3, 1, 3]])
#    f = np.array([[32, 64, 128], [16, 0, 1], [8, 4, 2]])
#    out, out2 = expand_zone(a, zone=1, win=(3,3))
    a_rc = reclass_vals(a,
                        old_vals=[1, 3, 5],
                        new_vals=[9, 5, 1],
                        mask=False,
                        mask_val=None)
    return a_rc


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
# https://stackoverflow.com/questions/47861214/
# using-numpy-as-strided-to-retrieve-subarrays-centered-on-main-diagonal
"""
theta = inclination of sun from 90 in radians
theta2 = slope angle
phi = ((450 - sun orientation from north in degrees) mod 360) * 180/pi
"""
