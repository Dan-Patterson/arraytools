# -*- coding: UTF-8 -*-
"""
:Script:   line_ang_azim.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2016-12-21
:
:Purpose:
:
:Functions:  help(<function name>) for help
:---------
: _demo  -  This function ...
:
:Notes:
: see help topic: np.info(np.arctan2)
: np.arctan2(dy, dx) is the format which differs from excel
: dx, dy - the differences in the respective coordinates x and y
: 360 = 2*np.pi, aka the circle in radians
:Results:
:-------                                 x-axis  compass azim
:  orig: [0, 0]: dest: [-1, 1]  line_dir:  135.0   NW    315
:  orig: [0, 0]: dest: [0, 1]   line_dir:   90.0   N       0, 360
:  orig: [0, 0]: dest: [1, 1]   line_dir:   45.0   NE     45
:  orig: [0, 0]: dest: [1, 0]   line_dir:    0.0   E      90
:  orig: [0, 0]: dest: [1, -1]  line_dir:  -45.0   SE    135
:  orig: [0, 0]: dest: [0, -1]  line_dir:  -90.0   S     180
:  orig: [0, 0]: dest: [-1, -1] line_dir: -135.0   SW    225
:  orig: [0, 0]: dest: [-1, 0]  line_dir:  180.0   W     270
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----

import sys
import numpy as np
from textwrap import dedent, indent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

# ---- functions ----


def line_dir(orig, dest, fromNorth=False):
    """Direction of a line given 2 points
    : orig, dest - two points representing the start and end of a line.
    : fromNorth - True or False gives angle relative to x-axis)
    :Notes:
    :
    """
    orig = np.asarray(orig)
    dest = np.asarray(dest)
    dx, dy = dest - orig
    ang = np.degrees(np.arctan2(dy, dx))
    if fromNorth:
        ang = np.mod((450.0 - ang), 360.)
    return ang


def demo(xc=0, yc=0, fromNorth=True):
    """ run the demo with the data below """
    p0 = np.array([xc, yc])  # origin point
    p1 = p0 + [-1, 1]   # NW
    p2 = p0 + [0, 1]    # N
    p3 = p0 + [1, 1]    # NE
    p4 = p0 + [1, 0]    # E
    p5 = p0 + [1, -1]   # SE
    p6 = p0 + [0, -1]   # S
    p7 = p0 + [-1, -1]  # SW
    p8 = p0 + [-1, 0]   # W
    #
    od = [[p0, p1], [p0, p2], [p0, p3], [p0, p4],
          [p0, p5], [p0, p6], [p0, p7], [p0, p8]]
    for pair in od:
        orig, dest = pair
        ang = line_dir(orig, dest, fromNorth=fromNorth)
        if fromNorth:
            dir = "From N."
        else:
            dir = "From x-axis"
        args = [orig, dest, dir, ang]
        print("orig: {}: dest: {!s:<8} {}: {!s:>6}".format(*args))


# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
    xc = 300000   # pick an origin x  0 or 300000 for example
    yc = 5025000  # pick an origin y  0 or 5025000
    demo(xc, yc, fromNorth=True)
