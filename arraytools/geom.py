# -*- coding: UTF-8 -*-
"""
:Script:   geom.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-10-20
:Purpose:  tools for working with numpy arrays
:Functions: a and b are arrays
: - e_area(a, b=None)
: - e_dist(a, b, metric='euclidean')
: - e_leng(a)
:References:  See ein_geom.py for full details and examples
:  https://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
:  https://iliauk.com/2016/03/02/centroids-and-centres-numpy-r/
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
# from fc import _xyID
# from tools import group_pnts

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=2, suppress=True,
                    threshold=10, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

_script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['_view_', '_reshape_', 'areas', 'center', 'centroid',  'e_area',
           'e_dist', 'e_leng', 'seg_lengths', 'total_length', 'lengths']


# ---- _view and _reshape_ are helper functions -----------------------------
def _view_(a):
    """return a view of the array using the dtype and length"""
    return a.view((a.dtype[0], len(a.dtype.names)))


def _reshape_(a):
    """Reshape arrays, structured or recarrays to a 2D ndarray.
    :Notes:
    :-----
    :(1) The length of the dtype is checked. Only object ('O') and arrays with
    :  a uniform dtype return 0.  Structured and recarrays will yield 1 or >.
    :
    :(2) dtypes are stripped and the array reshaped
    :  a = np.array([(341000., 5021000.), (341000., 5022000.),
    :                (342000., 5022000.), (341000., 5021000.)],
    :               dtype=[('X', '<f8'), ('Y', '<f8')])
    :  becomes...
    :  a = np.array([[  341000.,  5021000.], [  341000.,  5022000.],
    :                [  342000.,  5022000.], [  341000.,  5021000.]])
    :  a.dtype = dtype('float64')
    :
    :(3) 3D arrays are collapsed to 2D
    :  a.shape = (2, 5, 2) => np.product(a.shape[:-1], 2) => (10, 2)
    :
    :(4) Object arrays are processed object by object but assumed to be of a
    :  common dtype within, as would be expected from a gis package.
    """
    a = np.asanyarray(a)
    shp = len(a.shape)
    dt = len(a.dtype)
    if (dt == 0) and shp == 1:
        a = [_view_(i) for i in a]
        return a
    elif (dt == 0) and shp == 2:
        return a
    elif (dt == 0) and shp > 2:
        tmp = a.reshape(np.product(a.shape[:-1]), 2)
        return tmp.view('<f8')
    elif dt == 1:
        fld_name = a6.dtype.names[0]  # assumes 'Shape' field is the geometry
        return a[fld_name]
    elif dt >= 2:
        if shp == 1:
            if len(a) == a.shape[0]:
                a = _view_(a)
            else:
                a = [_view_(i) for i in a]
        else:
            a = [_view_(i) for i in a]
    return a


# ---- centers --------------------------------------------------------------
def center(a, remove_dup=True):
    """Return the center of an array. If the array represents a polygon, then
    :  a check is made for the duplicate first and last point to remove one.
    """
    if remove_dup:
        if np.all(a[0] == a[-1]):
            a = a[:-1]
    return a.mean(axis=0)


def centroid(a):
    """Return the centroid of a closed polygon
    : e_area required
    """
    x, y = a.T
    t = ((x[:-1] * y[1:]) - (y[:-1] * x[1:]))
    a_6 = e_area(a) * 6.0  # area * 6.0
    x_c = np.sum((x[:-1] + x[1:]) * t) / a_6
    y_c = np.sum((y[:-1] + y[1:]) * t) / a_6
    return np.asarray([-x_c, -y_c])


# ---- distance, length and area --------------------------------------------
# ----
def e_area(a, b=None):
    """Area calculation, using einsum.
    :  Some may consider this overkill, but consider a huge list of polygons,
    :  many multipart, many with holes and even multiple version therein.
    :Requires:
    :--------
    :  a - either a 2D+ array of coordinates or arrays of x, y values
    :  b - if a < 2D, then the y values need to be supplied
    :  Outer rings are ordered clockwise, inner holes are counter-clockwise
    :Notes: see ein_geom.py for examples
    :-----------------------------------------------------------------------
    """
    a = np.asanyarray(a)
    if b is None:
        xs = a[..., 0]
        ys = a[..., 1]
    else:
        xs, ys = a, b
    x0 = np.atleast_2d(xs[..., 1:])
    y0 = np.atleast_2d(ys[..., :-1])
    x1 = np.atleast_2d(xs[..., :-1])
    y1 = np.atleast_2d(ys[..., 1:])
    e0 = np.einsum('...ij,...ij->...i', x0, y0)
    e1 = np.einsum('...ij,...ij->...i', x1, y1)
    area = abs(np.sum((e0 - e1)*0.5))
    return area


def e_dist(a, b, metric='euclidean'):
    """Distance calculation for 1D, 2D and 3D points using einsum
    : a, b   - list, tuple, array in 1,2 or 3D form
    : metric - euclidean ('e','eu'...), sqeuclidean ('s','sq'...),
    :-----------------------------------------------------------------------
    """
    a = np.asarray(a)
    b = np.atleast_2d(b)
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 1:
        a = a.reshape(1, 1, a.shape[0])
    if a_dim >= 2:
        a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    if b_dim > 2:
        b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
    diff = a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr


def e_leng(a):
    """Length/distance between points in an array using einsum
    : Inputs
    :   a list/array coordinate pairs, with ndim = 3 and the
    :   Minimum shape = (1,2,2), eg. (1,4,2) for a single line of 4 pairs
    :   The minimum input needed is a pair, a sequence of pairs can be used.
    : Returns
    :   length - the total length/distance formed by the points
    :   d_leng - the distances between points forming the array
    :          - (40.0, [array([[ 10.,  10.,  10.,  10.]])])
    :-----------------------------------------------------------------------
    """
    #
#    d_leng = 0.0
    # ----
    def _cal(diff):
        """ perform the calculation
        :diff = g[:, :, 0:-1] - g[:, :, 1:]
        : for 4D
        " d = np.einsum('ijk..., ijk...->ijk...', diff, diff).flatten() or
        :   = np.einsum('ijkl, ijkl->ijk', diff, diff).flatten()
        : d = np.sum(np.sqrt(d)
        """
        d_leng = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff)).squeeze()
        length = np.sum(d_leng.flatten())
        return length, d_leng
    # ----
    diffs = []
    a = np.atleast_2d(a)
    if a.shape[0] == 1:
        return 0.0
    if a.ndim == 2:
        a = np.reshape(a, (1,) + a.shape)
    if a.ndim == 3:
        diff = a[:, 0:-1] - a[:, 1:]
        length, d_leng = _cal(diff)
        diffs.append(d_leng)
    if a.ndim == 4:
        length = 0.0
        for i in range(a.shape[0]):
            diff = a[i][:, 0:-1] - a[i][:, 1:]
            leng, d_leng = _cal(diff)
            diffs.append(d_leng)
            length += leng
    return length, diffs[0]


# ---- Batch calculations of e_area and e_leng ------------------------------
#
def areas(a):
    """Calls e_area to calculate areas for many types of nested objects.
    :  This would include object arrays, list of lists and similar constructs.
    :  Each part is considered separately.
    :Returns:
    :  A list with one or more areas.
    """
    a = np.asarray(a)
    if a.dtype == 'O':
        tmp = [_reshape_(i) for i in a]
        return [e_area(i) for i in tmp]
    if len(a.dtype) == 1:
        a = _reshape_(a)
    if len(a.dtype) > 1:
        a = _reshape_(a)
    if isinstance(a, (list, tuple)):
        return [e_area(i) for i in a]
    if a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    a_s = [e_area(i) for i in a]
    return a_s


def lengths(a, prn=False):
    """Calls e_leng to calculate lengths for many types of nested objects.
    :  This would include object arrays, list of lists and similar constructs.
    :  Each part is considered separately.
    :Returns:
    :  A list with one or more lengths. prn=True for optional printing
    """
    def _prn_(a_s):
        """optional result printing"""
        hdr = "{!s:<12}  {}\n".format("Tot. Length", "Seg. Length")
        r = ["{:12.3f}  {!r:}".format(*a_s[i]) for i in range(len(a_s))]
        print(hdr + "\n".join(r))
    #
    a = np.asarray(a)
    if a.dtype == 'O':
        tmp = [_reshape_(i) for i in a]
        a_s = [e_leng(i) for i in tmp]
        if prn:
            _prn_(a_s)
        return a_s
    if len(a.dtype) == 1:
        a = _reshape_(a)
    if len(a.dtype) > 1:
        a = _reshape_(a)
    if isinstance(a, (list, tuple)):
        return [e_leng(i) for i in a]
    if a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    a_s = [e_leng(i) for i in a]
    if prn:
        _prn_(a_s)
    return a_s


def total_length(a):
    """Just return total length from 'length' above
    :Returns: list of array(s) containing the total length for each object
    """
    a_s = lengths(a)
    result = [i[0] for i in a_s]
    return result


def seg_lengths(a):
    """Just return segment lengths from 'length above.
    :Returns: list of array(s) containing the segment lengths for each object
    """
    a_s = lengths(a)
    result = [i[1] for i in a_s]
    return result


# ---- Extras ----------------------------------------------------------------
def _samples_():
    """Sample arrays to test verious cases
    """
    a0 = np.array([[10., 10.], [10., 20.], [20., 20.], [10., 10.]])
    a1 = np.array([[20., 20.], [20., 30.], [30., 30.], [30., 20.], [20., 20.]])
    a2 = np.array([[30., 30.], [30., 40.], [40., 40.], [40., 30.], [30., 30.]])
    a3 = np.asarray([a1, a2])
    a4 = np.array([(10., 10.), (10., 20.), (20., 20.), (10., 10.)],
                  dtype=[('X', '<f8'), ('Y', '<f8')])
    a5 = np.array([(20., 20.), (20., 30.), (30., 30.), (30., 20.), (20., 20.)],
                  dtype=[('X', '<f8'), ('Y', '<f8')])
    a6 = np.array([([20.0, 20.0],), ([20.0, 30.0],), ([30.0, 30.0],),
                   ([30.0, 20.0],), ([20.0, 20.0],)],
                  dtype=[('Shape', '<f8', (2,))])
    a7 = np.array([([10.0, 10.0],), ([10.0, 20.0],), ([20.0, 20.0],),
                   ([10.0, 10.0],)],
                  dtype=[('Shape', '<f8', (2,))])
    a8 = np.asarray([a4, a4])
    a9 = np.asarray([a4, a5])
    a10 = np.asarray([a6, a6])
    a11 = np.asarray([a6, a7])
    a = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]
#    sze = [i.size for i in a]
#    shp = [len(i.shape) for i in a]
#    dt = [len(i.dtype) for i in a]
    return a


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
    from fc import _xyID
    from tools import group_pnts

    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = _samples_()
#    in_fc = r'C:\Git_Dan\a_Data\arcpytools_demo.gdb\Can_geom_sp_LCC'
    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\Carp_5x5km"   # full 25 polygons
    a = _xyID(in_fc)
    a_s = group_pnts(a, key_fld='IDs', keep_flds=['Xs', 'Ys'])
#    a_s = np.asarray(a_s)
#    a_area = areas(a_s)
#    a_tot_leng = total_length(a_s)
#    a_seg_leng = seg_lengths(a_s)
