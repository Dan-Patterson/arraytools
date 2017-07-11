# coding: utf-8
"""
Script:   array_plot_demo.py
Author:   Dan.Patterson@carleton.ca
Purpose:  To make a graphic representation of a n-dimensional array
Notes:
1  arr_3d:   Create a 3D array of a particular rows,cols shape
     (ie (Y,X)) and n layers in the form...
     ....  np.zeros( (n, rows, cols), dtype='int32')  ....
2  arr_2d:   Assign the values for each 'layer' in n to the array
3  plot_3d:  Determine the plot parameters.
4  arr_grid: plot the grid using the range of values in each dimension
     to ensure consistency in tone/colour of the resultant plot
5  The __main__ section calls def main(...) should one wish to change
   the parameters.
Tips:
For ax.matshow, there are many other options:
  interpolation: 'none','nearest'
  cmap: cm.gray, cm.gray_r, cm.cool, cm.cool_r, cm.coolwarm
        cm.coolwarm_r, cm.gist_heat and cm.gist_heat_r

My thread on formatting: 2016-03-22
    http://stackoverflow.com/questions/36145179/numpy-array-representation-and-formatting

hpaulj's solution:
  arr1=np.arange(2*3*4).reshape(3,4,2)
  z=[np.array2string(i).splitlines() for i in arr1]
  print '\n'.join(['\t'.join(k) for k in zip(*alist)])
>>> print('\n'.join(['\t'.join(k) for k in zip(*z)]))
[[ 0  1  2]	[[ 0  1  2]	[[ 0  1  2]
 [ 3  4  5]	 [ 3  4  5]	 [ 3  4  5]
 [ 6  7  8]	 [ 6  7  8]	 [ 6  7  8]
 [ 9 10 11]	 [ 9 10 11]	 [ 9 10 11]
 [12 13 14]]	 [12 13 14]]	 [12 13 14]]
ie
z=[np.array2string(i).splitlines() for i in a_3d]

>>> aa
array([[[   0,    1,    2],
        [   3,    4,    5],
        [   6,    7,    8],
        [   9,   10,   11],
        [  12,   13,   14]],

       [[   0,   10,   20],
        [  30,   40,   50],
        [  60,   70,   80],
        [  90,  100,  110],
        [ 120,  130,  140]],

       [[   0,  100,  200],
        [ 300,  400,  500],
        [ 600,  700,  800],
        [ 900, 1000, 1100],
        [1200, 1300, 1400]]], dtype=int32)
>>> z = [np.array2string(i).splitlines() for i in aa]
>>> print('\n'.join(['\t'.join(k) for k in zip(*z)]))
[[ 0  1  2]	[[  0  10  20]	[[   0  100  200]
 [ 3  4  5]	 [ 30  40  50]	 [ 300  400  500]
 [ 6  7  8]	 [ 60  70  80]	 [ 600  700  800]
 [ 9 10 11]	 [ 90 100 110]	 [ 900 1000 1100]
 [12 13 14]]	 [120 130 140]]	 [1200 1300 1400]]

aa[1] = aa[0]+5 aa[2] = aa[0]+10
>>> z = [np.array2string(i).splitlines() for i in aa]
>>> print('\n'.join(['\t'.join(k) for k in zip(*z)]))
[[ 0  1  2]	[[ 5  6  7]	[[10 11 12]
 [ 3  4  5]	 [ 8  9 10]	 [13 14 15]
 [ 6  7  8]	 [11 12 13]	 [16 17 18]
 [ 9 10 11]	 [14 15 16]	 [19 20 21]
 [12 13 14]]	 [17 18 19]]	 [22 23 24]]

See ... to_row(a) ... for formatted output using set_printoptions too

to_row(a_3d) from post...

Array...  3D  r  c
  shape: (3, 10, 3) size: 90
a[0].......	a[1].......	a[2].......
[[ 9  8 13]	[[11 12 14]	[[12  9 12]
 [ 4 11 11]	 [ 5  2 12]	 [11  5  8]
 [ 6 12 14]	 [ 6  7  3]	 [ 2 12  5]
 .......... 	 .......... 	 ..........
 [ 5 13  8]	 [ 3  6  0]	 [ 8  0  2]
 [ 6 10  5]	 [ 2 14 11]	 [ 2 11  4]
 [ 8  0  4]]	 [ 7  9 12]]	 [ 7  2 14]]
Note:
- array2String is from numpy.core.arrayprint
- array2string(a, prefix="", separator=" ")
- np.array2string(a_3d, prefix = "**") # prefix is length of the
                    prefix normally "" nothing len = 0
- np.array2string(a_3d, prefix = "", separator=",") # sep is normally " " space
np.array2string.func_defaults
(None, None, None, ' ', '', <built-in function repr>, None)

import textwrap
    added shorten and indent in 3.3/3.4
Tabulate ... code from github
   https://github.com/gregbanks/python-tabulate/blob/master/tabulate.py
"""
import sys
import numpy as np
from textwrap import dedent
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mc

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def arr_3d(n=1, rows=1, cols=1, dt='int32'):
    """Create a 3D array with shape (n, rows, cols)
       z = np.zeros((2,5,3),dtype='int32') for example.
    """
    z = np.zeros((n, rows, cols), dtype=dt)
    return z


def arr_2d(rows=1, cols=1, zmin=1, zmax=5, rand_nums=True):
    """Create a 2D array with shape(rows, cols) of integer dtype"""
    if rand_nums:
        a = np.random.randint(low=zmin, high=zmax, size=(rows, cols))
    else:
        a = np.arange(rows*cols).reshape((rows, cols))
    return a


def plot_3d(rows=1, cols=1):
    """Set the parameters of the 3D array for plotting"""
    x = np.arange(cols)
    y = np.arange(rows)
    xg = np.arange(-0.5, cols + 0.5, 1)
    yg = np.arange(-0.5, rows + 0.5, 1)
    return [x, y, xg, yg]


def arr_grid(arr):
    """Demo:  emulate a 3D array plotting the first then adding
       the others the next time around.
    """
    n, rows, cols = arr.shape
    m_min = arr.min()
    m_max = arr.max()
    result = plot_3d(rows, cols)
    x, y, xg, yg = result
    fig, axes = plt.subplots(1, n, sharex=True, sharey=True)
    idx = 0
    fig.set_facecolor('w')
    for ax in axes:
        a = arr[idx]
        col_lbl = "Cols: for " + str(idx)
        ax.set_aspect('equal')
        ax.set_adjustable('box-forced')  # prevents spaces
        ax.set_xticks(xg, minor=True)
        ax.set_yticks(yg, minor=True)
        ax.set_xlabel(col_lbl, labelpad=12)
        ax.xaxis.label_position = 'top'
        ax.xaxis.label.set_fontsize(12)
        if idx == 0:
            ax.set_ylabel("Rows", labelpad=2)  # was 12
            ax.yaxis.label.set_fontsize(12)
        a_min = a[idx].min()
        a_max = a[idx].max()
        ax.grid(which='minor', axis='x', linewidth=1, linestyle='-', color='k')
        ax.grid(which='minor', axis='y', linewidth=1, linestyle='-', color='k')
        # print("idx {}\n{}".format(idx,a[idx]))
        t = [[x, y, a[y, x]]
             for y in range(rows)
             for x in range(cols)]
        for i, (x_val, y_val, c) in enumerate(t):
            ax.text(x_val, y_val, c, va='center', ha='center', fontsize=12)
        ax.matshow(arr[idx], cmap=cm.gray_r, interpolation='nearest',
                   vmin=m_min, vmax=m_max, alpha=0.2)
        idx += 1
    plt.show()
    # plt.close()
    return plt, fig, ax, arr


def de_line(a):
    """ remove those pesky extra lines"""
    a = a.squeeze()
    idx = np.arange(1, a.shape[0])
    a_s = str(a)
    rep_list = [["\n\n\n", "\na[{}]....\n"],
                ["\n\n", "\na[{}]....\n"]
                ]
    for rep in rep_list:
        a_s = a_s.replace(rep[0], rep[1])
    a_s = a_s.format(*list(idx))
    frmt = "array...\nshape {} ndim {} size {}\na[0]...\n"
    a_s = frmt.format(a.shape, a.ndim, a.size) + a_s
    return a_s


def array2row(arr, with_hdr=True):
    """Calling function"""
    po = np.get_printoptions()
    dim = arr.ndim
    shp = arr.ndim
    zz_final = ""
    if dim == 4:
        n = arr.shape[0]
        frmt = '\nArray... {}D  shape: {} size: {}\n3D subarrays: {}\n'
        hdr = frmt.format(arr.ndim, arr.shape, arr.size, n)
        zz_final += hdr
        for i in range(n):
            a = arr[i]
            zz_final += to_row(a, with_hdr=with_hdr)
        np.set_printoptions(po)
        return zz_final, arr
    #
    if dim == 3:
        a = arr[:]
    elif dim == 2:
        shp = (2,) + shp
        dt = arr.dtype
        a = np.zeros(shp, dt)
        a[0] = arr
    else:
        return zz, arr
    zz_final += "\n" + to_row(a, with_hdr=with_hdr)
    np.set_printoptions(po)
    return zz_final, arr


def to_row(a, with_hdr=True):
    """Reformat 3D arrays to rows.  Arrays with different dimensions
    :   are reformatted to 3D arrays.
    : - Convert array to a string and split into lines
    : - Join the strings by zipping and separate with tab
    : - Determine the width of the first row's, first element
    : - This will be used to replace ... with some padding feature
    : - see ell and zz lines
    """
    if a.ndim < 3:
        return a
    shp = a.shape
    n = shp[0]
    n_s = np.arange(n).tolist()
    np.set_printoptions(threshold=(n*10 + 1))
    hdr = ""
    #
    z = [np.array2string(i).splitlines() for i in a]
    z1 = [[(j.replace("[[", "[")).replace("]]", "]").strip()
          for j in i] for i in z]
    zz = '\n'.join(['  '.join(k) for k in zip(*z1)])
    ell = len(z1[0][0])
    zz = zz.replace("...,", "."*(ell))
    if with_hdr:
        frmt = '\nArray...  {}D   r  c\n  shape: {} size: {} \n'
        hdr = frmt.format(a.ndim, shp, a.size)
        h = 'a[{}]' + "."*(ell-4)
        hdr = hdr + ''.join([h.format(i) + '  ' for i in n_s])
    zz = hdr + "\n" + zz + "\n"
    return zz


def make_arr(shp=(2, 5, 3), zmin=1, zmax=5, dt='int32', rand_nums=True):
    """Run a demo creating a representation of a 3D array"""
    n, r, c = shp
    a_3d = arr_3d(n, r, c, dt)
    for i in range(n):
        a_3d[i] = arr_2d(r, c, zmin, zmax, rand_nums)
    return a_3d


def demo_3d(a_3d):
    """restructure a 3d array"""
    frmt = """
    : Test various array representations using a combination of set_print
    : options and array formats.
    :  np.set_printoptions(edgeitems=3, linewidth=80,
    :                      precision=2, threshold=20)")
    \nThe array... (a_3d) ...\n{}
    """
    print(dedent(frmt).format(a_3d))
    a_de = de_line(a_3d)
    print("\nde_line(a_3d) with extra info...\n{}".format(a_de))
    zz = to_row(a_3d)
    print("\nto_row(a_3d) from post...\n{}".format(zz))
    return a_3d, zz


def demo_4d(a_3d, n=5):
    """ demo of 4D arrays"""
    zz, a_3d = array2row(a_3d, with_hdr=True)
    shp = (n,) + a_3d.shape
    dt = a_3d.dtype
    a_4d = np.zeros(shp, dt)
    a_4d[0] = a_3d
    for i in range(1, n):
        a_4d[i] = a_4d[0] + i*2
    zz4, a_4d = array2row(a_4d, with_hdr=True)
    print(zz4)
    return a_4d, zz4


if __name__ == "__main__":
    """ array reformatting and array as plot demo
        Create the array, show its optional 3D and 4D configurations.
        Optional parameters in make_arr allow you to configure xhape
        values range, dat type and whether random or sequential values
        are created.
    """
    # Need a_3d for the demos to run.  Comment out the lines you
    # don't want. Parameters returned for further analysis.
    #
    a_3d = make_arr(shp=(3, 6, 3), zmin=0, zmax=15,
                    dt='int32', rand_nums=True)
    # arr_grid .... plotting demo
#    plt, fig, ax, a_3d = arr_grid(a_3d)
    # demo_3d .... 3D array output formats
    a_3d, zz = demo_3d(a_3d)
    # demo_4d .... 4D array output formats
    a_4d, zz4 = demo_4d(a_3d, n=5)
