# -*- coding: UTF-8 -*-
"""
===
tbl
===

Script :   tbl.py   tools for working with text arrays in table form

Author :   Dan.Patterson@carleton.ca

Modified : 2019-02-11

Purpose :  Tabulate data

- Unique counts on 2 or more variables.
- Sums, mins, max etc on variable classes

Requires
--------
`frmts.py` is required since it uses print functions from there

`prn` is used for fancy printing if it loads correctly

Notes
-----
To convert esri geodatabase tables or shapefile tables to arrays, use the
following guidelines::

    >>> float_min = np.finfo(np.float).min
    >>> float_max = np.finfo(np.float).max
    >>> int_min = np.iinfo(np.int_).min
    >>> int_max = np.iinfo(np.int_).max
    >>> f = 'C:/some/path/your.gdb/your_featureclass'
    >>> null_dict = {'Int_fld': int_min, 'Float_fld': float_min}  # None stuff
    >>> flds = ['Int_field', 'Txt01', 'Txt02']  # 2 text fields
    >>> a = arcpy.da.TableToNumPyArray(in_table=f, field_names=flds,
                                      skip_nulls=False,
                                      null_value=null_dict)  # if needed
    >>> row = 'Txt01'
    >>> col = 'Txt02'
    >>> ctab = crosstab(a, row, col, verbose=False)

**Useful tips**

`...install folder.../Lib/site-packages/numpy/core/numerictypes.py`

>>> # import string is costly to import!
>>> # Construct the translation tables directly
>>> # A = chr(65), a = chr(97)
>>> _all_chars = [chr(_m) for _m in range(256)]
>>> _ascii_upper = _all_chars[65:65+26]
>>> _ascii_lower = _all_chars[97:97+26]
>>> _just_numbers = _all_chars[48:58]
>>> LOWER_TABLE = ''.join(_all_chars[:65] + _ascii_lower + _all_chars[65+26:])
>>> UPPER_TABLE = ''.join(_all_chars[:97] + _ascii_upper + _all_chars[97+26:])

- np.char.split(s, ' ')
- np.char.startswith(s, 'S')
- np.char.strip()
- np.char.str_len(s)

- np.sum(np.char.startswith(s, ' '))  # check for leading spaces
- np.sum(np.char.endswith(s0, ' '))   # check for trailing spaces
- s0 = np.char.rstrip(s0)

Partitioning::

    lp = np.char.partition(s0, ' ')[:, 0]   # get the left-most partition
    rp = np.char.rpartition(s0, ' ')[:, -1] # get the right-most partition
    lpu, lpcnts= np.unique(lp, return_counts=True)
    rpu, rpcnts= np.unique(rp, return_counts=True)

References
----------

`<https://stackoverflow.com/questions/12983067/how-to-find-unique-vectors-of
-a-2d-array-over-a-particular-axis-in-a-vectorized>`_.

`<https://stackoverflow.com/questions/16970982/find-unique-rows-
in-numpy-array>`_.

`<http://stackoverflow.com/questions/38030054/create-adjacency-matrix-in-
python-for-large-dataset>`_.

np.unique - in the newer version, they use flags to get the sums

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
from textwrap import dedent
import numpy as np

# ---- others from above , de_punc, _describe, fc_info, fld_info, null_dict,

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

if 'prn' not in locals().keys():
    try:
        from arraytools.frmts import prn
    except:
        prn = print

__all__ = ['find_a_in_b',
           'find_in',
           '_split_sort_slice_',
           'tbl_count',
           'tbl_sum']

# ---- text columns... via char arrays
#
#def tbl_replace(a, col=None, from_=None, to_=None):
#    """table replace
#    """
#    #np.char.replace(
#    pass
#    return None

def find_a_in_b(a, b, a_fields=None, b_fields=None):
    """Find the indices of the elements in a smaller 2d array contained in
    a larger 2d array. If the arrays are stuctured with field names,then these
    need to be specified.  It should go without saying that the dtypes need to
    be the same.

    Parameters
    ----------
    a, b : 1D or 2D, ndarray or structured/record arrays
        The arrays are arranged so that `a` is the smallest and `b` is the
        largest.  If the arrays are stuctured with field names, then these
        need to be specified.  It should go without saying that the dtypes
        need to be the same.
    a_fields, b_fields : list of field names
        If the dtype has names, specify these in a list.  Both do not need
        names.

    Examples
    --------
    To demonstrate, a small array was made from the last 10 records of a larger
    array to check that they could be found.

    >>> a.dtype
    ``([('ID', '<i4'), ('X', '<f8'), ('Y', '<f8'), ('Z', '<f8')])``
    >>> b.dtype ``([('X', '<f8'), ('Y', '<f8')])``
    >>> a.shape, b.shape # ((69688,), (10,))
    >>> find_a_in_b(a, b, flds, flds)
    array([69678, 69679, 69680, 69681, 69682,
           69683, 69684, 69685, 69686, 69687], dtype=int64)

    References
    ----------
    This is a function from the arraytools.tbl module
    `<https://stackoverflow.com/questions/38674027/find-the-row-indexes-of-
    several-values-in-a-numpy-array/38674038#38674038>`_.
    """
    def _view_(a):
        """from the same name in arraytools"""
        return a.view((a.dtype[0], len(a.dtype.names)))
    #
    small, big = [a, b]
    if a.size > b.size:
        small, big = [b, a]
    if a_fields is not None:
        small = small[a_fields]
        small = _view_(small)
    if b_fields is not None:
        big = big[b_fields]
        big = _view_(big)
    if a.ndim >= 1:  # last slice, if  [:2] instead, it returns both indices
        indices = np.where((big == small[:, None]).all(-1))[1]
    return indices, big, small


def find_in(a, col, what, where='in', any_case=True, pull='all'):
    """Query a recarray/structured array for values

    Parameters
    ----------
    a : recarray/structured array
        Only text columns can be queried
    col : column/field to query
        Only 1 field can be queried at a time for the condition.
    what : string or number
        The query.  If a number, the field is temporarily converted to a
        text representation for the query.
    where : string
        s, i, eq, en .... `st`(arts with), `in`, `eq`(ual), `en`(ds with)
    any_case : boolean
        True, will find records regardless of `case`, applies to text fields
    extract: text or list
        - `all`,  extracts all records where the column case is found
        - `list`, extracts the records for only those fields in the list
    Example
    -------
    >>> find_text(a, col=`FULLNAME`, what=`ABBEY`, pull=a.dtype.names[:2])
    """
    # ---- error checking section ----
    e0 = """
    Query error: You provided...
    dtype: {}  col: {} what: {}  where: {}  any_case: {}  extract: {}
    Required...\n{}
    """
    if a is None:
        return a
    err1 = "\nField not found:\nQuery fields: {}\nArray fields: {}"
    errors = [a.dtype.names is None,
              col is None, what is None,
              where.lower()[:2] not in ('en', 'eq', 'in', 'st'),
              col not in a.dtype.names]
    if sum(errors) > 0:
        arg = [a.dtype.kind, col, what, where, any_case, pull, find_in.__doc__]
        print(dedent(e0).format(*arg))
        return None
    if isinstance(pull, (list, tuple)):
        names = a.dtype.names
        r = [i in names for i in pull]
        if sum(r) != len(r):
            print(err1.format(pull, names))
            return None
    # ---- query section
    # convert column values and query to lowercase, if text, then query
    c = a[col]
    if c.dtype.kind in ('i', 'f', 'c'):
        c = c.astype('U')
        what = str(what)
    elif any_case:
        c = np.char.lower(c)
        what = what.lower()
    where = where.lower()[0]
    if where == 'i':
        q = np.char.find(c, what) >= 0   # ---- is in query ----
    elif where == 's':
        q = np.char.startswith(c, what)  # ---- startswith query ----
    elif where == 'eq':
        q = np.char.equal(c, what)
    elif where == 'en':
        q = np.char.endswith(c, what)    # ---- endswith query ----
    if q.sum() == 0:
        print("none found")
        return None
    if pull == 'all':
        return a[q]
    pull = np.unique([col] + list(pull))
    return a[q][pull]

def group_sort(a, group_fld, sort_fld, ascend=True):
    """Group records in an structured array and sort on the sort_field.  The
    order of the grouping field will be in ascending order, but the order of
    the sort_fld can sort internally within the group.

    Parameters
    ----------
    a : structured/recarray
        Array must have field names to enable splitting on and sorting by
    group_fld : text
        The field/name in the dtype used to identify groupings of features
    sort_fld : text
        As above, but this field contains the numeric values that you want to
        sort by.
    ascend : boolean
        **True**, sorts in ascending order, so you can slice for the lowest
        `num` records. **False**, sorts in descending order if you want to
        slice the top `num` records

    Example
    -------
    >>> fn = "C:/Git_Dan/arraytools/Data/pnts_in_poly.npy"
    >>> a = np.load(fn)
    >>> out = _split_sort_slice_(a, split_fld='Grid_codes', val_fld='Norm')
    >>> arcpy.da.NumPyArrayToFeatureClass(out, out_fc, ['Xs', 'Ys'], '2951')

    References
    ----------
    `<https://community.esri.com/blogs/dan_patterson/2019/01/29/split-sort-
    slice-the-top-x-in-y>`_.

    `<https://community.esri.com/thread/227915-how-to-extract-top-five-max-
    points>`_
    """
    ordered = _split_sort_slice_(a, split_fld=group_fld, order_fld=sort_fld)
    final = []
    if ascend:
        for r in ordered:
            final.extend(r)
    else:
        for r in ordered:
            r = r[::-1]
            final.extend(r)
    return np.asarray(final)


def n_largest_vals(a, group_fld=None, val_fld=None, num=1):
    """Run `split_sort_slice` to get the N largest values in the 
    """
    ordered = _split_sort_slice_(a, split_fld=group_fld, order_fld=val_fld)
    final = []
    for r in ordered:
        r = r[::-1]
        num = min(num, r.size)
        final.extend(r[:num])
    return np.asarray(final)


def n_smallest_vals(a, group_fld=None, val_fld=None, num=1):
    """Run `split_sort_slice` to get the N smallest values in the 
    """
    ordered = _split_sort_slice_(a, split_fld=group_fld, order_fld=val_fld)
    final = []
    for r in ordered:
        num = min(num, r.size)
        final.extend(r[:num])
    return np.asarray(final)


def _split_sort_slice_(a, split_fld=None, order_fld=None):
    """Split a structured array into groups of common values based on the
    split_fld, key field.  Once the array is split, the array is sorted on a
    val_fld and sliced for the largest or smallest `num` records.

    See Also
    --------
    Documentation is shown in `group_sort`

    """
    def _split_(a, fld):
        """split unsorted array"""
        out = []
        uni, _ = np.unique(a[fld], True)
        for _, j in enumerate(uni):
            key = (a[fld] == j)
            out.append(a[key])
        return out
    #
    err_0 = """
    A structured/recarray with a split_field and a order_fld is required.
    You provided\n    array type  : {}"""
    err_1 = """
    split_field : {}
    order field : {}  
    """
    if a.dtype.names is None:
        print(err_0.format(type(a)))
        return a
    check = sum([i in a.dtype.names for i in [split_fld, order_fld]])
    if check != 2:
        print((err_0 + err_1).format(type(a), split_fld, order_fld))
        return a
    #
    subs = _split_(a, split_fld)
    ordered = []
    for _, sub in enumerate(subs):
        r = sub[np.argsort(sub, order=order_fld)]
        ordered.append(r)
    return ordered


def tbl_count(a, row=None, col=None, verbose=False):
    """Crosstabulate 2 fields data arrays, shape (N,), using np.unique.
    scipy.sparse has similar functionality and is faster for large arrays.

    Parameters
    ----------
    a : array
        A 2D array of data with shape(N,) representing two variables.
    row : field/column
        The table column/field to use for the row variable
    col : field/column
        The table column/field to use for thecolumn variable

    Notes
    -----
    See useage section above for converting Arc* tables to arrays.

    Returns
    -------
    ctab : array
        the crosstabulation result as row, col, count array
    a : array
        the crosstabulation in a row, col, count, but filled out whether a
        particular combination exists or not.
    r, c : names
        unique values/names for the row and column variables
    """
    names = a.dtype.names
    assert row in names, "The.. {} ..column, not found in array.".format(row)
    assert col in names, "The.. {} ..column, not found in array.".format(col)
    r_vals = a[row]
    c_vals = a[col]
    dt = np.dtype([(row, r_vals.dtype), (col, c_vals.dtype)])
    rc = np.asarray(list(zip(r_vals, c_vals)), dtype=dt)
    u, cnt = np.unique(rc, return_counts=True)
    rcc_dt = u.dtype.descr
    rcc_dt.append(('Count', '<i4'))
    ctab = np.asarray(list(zip(u[row], u[col], cnt)), dtype=rcc_dt)
    if verbose:
        prn(ctab)
    else:
        return ctab


def tbl_sum(a, row=None, col=None, val_fld=None):
    """Tabular sum of values for two attributes

    Parameters
    ----------
    a : array
        Structured/recarray
    row, col : string
        The fields to be used as the table rows and columns
    val_fld : string
        The field that will be summed for the unique combinations of
        row/column classes

    Returns
    -------
    A table summarizing the sums for the row/column combinations.
    """
    # ---- Slice the input array using the row/column fields, determine the
    # unique combinations of their attributes.  Create the output dtype
    names = a.dtype.names
    assert row in names, "The.. {} ..column, not found in array.".format(row)
    assert col in names, "The.. {} ..column, not found in array.".format(col)
    val_kind = a[val_fld].dtype.kind
    if val_kind not in ('i', 'f'):
        print("\nThe value field must be numeric")
        return None
    if val_kind == 'f':
        val_type = '<f8'
    elif val_kind == 'i':
        val_type = '<i4'
    rc = a[[row, col]]
    sum_name = val_fld +'_sum'
    dt = rc.dtype.descr + [(sum_name, val_type)]
    uniq = np.unique(rc)
    #
    # ----
    out_ = []
    for u in uniq:
        c0, c1 = u
        idx = np.logical_and(a[row] == c0, a[col] == c1)
        val = np.nansum(a[val_fld][idx])
        out_.append([c0, c1, val])
    out_ = np.array(out_)
    z = np.empty((len(out_),), dtype=dt)
    z[row] = out_[:, 0]
    z[col] = out_[:, 1]
    z[sum_name] = out_[:, 2].astype(val_kind)
    return z


# ---- crosstab from tool, uncomment for testing or tool use
def _demo():
    """Load the sample file for testing
    """
    # script = sys.argv[0]  # the script path defined earlier
    in_tbl = script.rpartition("/")[0] + '/Data/sample_20.npy'  # sample_20.npy
    a = np.load(in_tbl)
    ctab = tbl_count(a, row='County', col='Town', verbose=True)
    return a, ctab

def _data():
    """base file"""
    in_tbl = script.rpartition("/")[0] + '/Data/points_2000.npy'
    a = np.load(in_tbl)
    return a

if __name__ == "__main__":
    # print the script source name.
    print("Script... {}".format(script))
#    ctab, counts, out_tbl = tab_count(a['County'], a['Town'],
#    r_fld='County', c_fld='Town', verbose=False)
#    ctab, a, result, r, c = _demo()
