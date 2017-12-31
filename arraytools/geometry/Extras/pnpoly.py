# note comes from here
# https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html

cimport cython
import numpy as np
cimport numpy as np

dummy_name = "foo"

DTYPE_FLOAT64 = np.float64
ctypedef np.float64_t DTYPE_FLOAT64_t
DTYPE_BOOL = np.bool
ctypedef np.uint8_t DTYPE_BOOL_t

@cython.boundscheck(False)
@cython.wraparound(False)

def pnpoly(vert,test):

    """
    Determine whether m test points are within a polygon defined by a set of 
    n vertices.

    Adapted from the code pnpoly.c by W. Randolph Franklin
    http://www.ecse.rpi.edu/~wrf/Research/Short_Notes/pnpoly.html

    """
 
    cdef int i
    cdef int j
    cdef int k
    cdef int m
    cdef int n
    cdef np.ndarray[np.uint8_t,ndim=1,cast=True] result
   
    n = vert.shape[0]
    m = test.shape[0]

    result = np.zeros(m,DTYPE_BOOL)

    for i in range(n):
        j = (i+n-1) % n
        for k in range(m):
           if(((vert[i,1] > test[k,1]) != (vert[j,1] > test[k,1])) and \
               (test[k,0] < (vert[j,0]-vert[i,0]) * (test[k,1]-vert[i,1])/ \
               (vert[j,1]-vert[i,1]) + vert[i,0]) ):
               result[k] = not result[k]

    return result
    

