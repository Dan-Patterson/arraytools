 # -*- coding: UTF-8 -*-
"""
:Script:   crosstab.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-09-23
:Purpose:  Crosstabulate data
:Notes:
:
:References:
: https://stackoverflow.com/questions/12983067/how-to-find-unique-vectors-of
:         -a-2d-array-over-a-particular-axis-in-a-vectorized
: https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
: http://stackoverflow.com/questions/38030054/
:      create-adjacency-matrix-in-python-for-large-dataset
: np.unique
: in the newer version, they use flags to get the sums
:
"""
import sys
import numpy as np
import arcpy
from textwrap import dedent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=50, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]


def _prn(r, c, a):
    """fancy print formatting.
    """
    r_sze = max(max([len(i) for i in r]), 5)
    c_sze = [max(len(str(i)), 5) for i in c]
    f_0 = '{{!s:<{}}} '.format(r_sze)
    f_1 = ('{{!s:>{}}} '*len(c)).format(*c_sze)
    frmt = f_0 + f_1
    hdr = '-'*r_sze
    print(frmt.format(hdr, *c))
    for i in range(len(r)):
        print(frmt.format(r[i], *a[i]))


def crosstab(row, col, verbose=False):
    """Crosstabulate 2 data arrays, shape (N,), using np.unique.
    :  scipy.sparse has similar functionality and is faster for large arrays.
    :
    :Requires:  A 2D array of data with shape(N,) representing two variables
    :--------
    : row - row variable
    : col - column variable
    :
    :Returns:
    : ctab - the crosstabulation result as row, col, count array
    : a - the crosstabulation in a row, col, count, but filled out whether a
    :     particular combination exists or not.
    : r, c - unique values/names for the row and column variables
    :
    """
    dt = np.dtype([('row', row.dtype), ('col', col.dtype)])
    rc = np.asarray(list(zip(row, col)), dtype=dt)
    r = np.unique(row)
    c = np.unique(col)
    u, idx, inv, cnt = np.unique(rc, return_index=True, return_inverse=True,
                                 return_counts=True)
    rcc_dt = u.dtype.descr
    rcc_dt.append(('Count', '<i4'))
    ctab = np.asarray(list(zip(u['row'], u['col'], cnt)), dtype=rcc_dt)
    z = np.zeros((len(r), len(c)), dtype=np.int_)
    rc = [[(np.where(r == i[0])[0]).item(),
           (np.where(c == i[1])[0]).item()] for i in ctab]
    for i in range(len(ctab)):
        rr, cc = rc[i]
        z[rr, cc] = ctab[i][2]
    if verbose:
        _prn(r, c, a)
    return ctab, z, r, c


def _demo():
    """run a test using a file
    : TableToNumPyArray(in_table, field_names, {where_clause},
    :                   {skip_nulls}, {null_value})
    """
    f = r'C:\FPA_2\D_gdb_files\FPA_final_Sept_13.gdb\FPA_final_Sept_13_sorted'
    flds = [i.name for i in arcpy.ListFields(f)]
    null_dict = {'OBJECTID': -9, 'Student_num': -9, 'Code': -9, 'Year_': -9}
    t = arcpy.da.TableToNumPyArray(in_table=f, field_names=flds,
                                   skip_nulls=False, null_value=null_dict)
#    rows = t['UG1']  # for undergrad
    rows = t['Grad1']  # for grad
    cols = t['Sector']
    ctab, a, r, c = crosstab(rows, cols, verbose=True)
    return ctab, a, r, c


if __name__ == "__main__":
    """run crosstabulation with data"""
    ctab, a, r, c = _demo()
