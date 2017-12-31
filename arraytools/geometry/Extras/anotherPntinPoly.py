# -*- coding: UTF-8 -*-
"""
:Script:   .py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-xx-xx
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np


ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def on_negative_side(p, v1, v2):
    d = v2 - v1
    return np.dot(np.array([-d[1], d[0]]), p - v1) < 0


def in_side(p, v1, v2, n):
    if n <= 0:
        return False
    d = v2 - v1
    l = np.linalg.norm(d)
    s = np.dot(d / l, p - v1)
    if s < 0 or s > l:  # No need for a check if point is outside edge 'boundaries'
        return False
    # Yves's check
    nd = np.array([-d[1], d[0]])
    m_v = nd * np.sqrt(3) / 6
    if np.dot(nd / l, v1 - p) > np.linalg.norm(m_v):
        return False
    # Create next points
    p1 = v1 + d/3
    p2 = v1 + d/2 - m_v
    p3 = v1 + 2*d/3
    # Check with two inner edges
    if on_negative_side(p, p1, p2):
        return in_side(p, v1, p1, n-1) or in_side(p, p1, p2, n-1)
    if on_negative_side(p, p2, p3):
        return in_side(p, p2, p3, n-1) or in_side(p, p3, v2, n-1)
    return True


def _in_koch(p, V, n):
    V_next = np.concatenate((V[1:], V[:1]))
    return all(not on_negative_side(p, v1, v2) or in_side(p, v1, v2, n)
               for v1, v2 in zip(V, V_next))


def in_koch(L, V, n):
    # Triangle points (V) are positive oriented
    return [p for p in L if _in_koch(p, V, n)]


L = np.array([(16, -16), (90, 90), (40, -40), (40, -95), (50, 10), (40, 15)])
V = np.array([(0, 0), (50, -50*np.sqrt(3)), (100, 0)])
for n in range(3):
    print(n, in_koch(L, V, n))
print(in_koch(L, V, 100))

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
    #_demo()

