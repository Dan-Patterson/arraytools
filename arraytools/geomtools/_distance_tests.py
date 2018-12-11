# -*- coding: utf-8 -*-
"""

=======

Script :   .py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-

Purpose :  tools for working with numpy arrays and geometry

Results:
--------
>>> a = np.random.rand(2, n, 3)  # 10, 100, 1000 
::
    1 linalg_norm
    2 sqrt_sum
    3 scipy_distance
    4 sqrt_einsum(
                          time
    method   1      2      3         4
    n = 10   6.25   5.64   94.5     3.18  µs
       100   8.57   7.65  938.0     4.25  µs
      1000  30.5   29.1    9.27 ms 14.0
     10000 204    195     93 ms    71.6 
References:
-----------
`<https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance
-be-calculated-with-numpy>`_.
"""
import sys
import numpy as np
from scipy.spatial import distance


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def linalg_norm(data):
    a, b = data
    return np.linalg.norm(a-b, axis=1)


def sqrt_sum(data):
    a, b = data
    return np.sqrt(np.sum((a-b)**2, axis=1))


def scipy_distance(data):
    a, b = data
    return list(map(distance.euclidean, a, b))


def sqrt_einsum(data):
    a, b = data
    a_min_b = a - b
    return np.sqrt(np.einsum('ij,ij->i', a_min_b, a_min_b))

np.random.RandomState(123)

n = 10
a = np.random.rand(2, n, 3)

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """

