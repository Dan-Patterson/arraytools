# -*- coding: UTF-8 -*-
"""
tbl
===

Script :   tbl.py   tools for working with text array in table form

Author :   Dan.Patterson@carleton.ca

Modified : 2018-10-27

Purpose :  Tabulate data
    Unique counts on 2 or more variables.
    Sums, mins, max etc on variable classes

Notes:
------
Useful tip:

`<C:\ArcGISPro\bin\Python\envs\arcgispro-py3\Lib\site-packages\
numpy\core\numerictypes.py>`_.

>>> # "import string" is costly to import!
>>> # Construct the translation tables directly
>>> #   "A" = chr(65), "a" = chr(97)
>>> _all_chars = [chr(_m) for _m in range(256)]
>>> _ascii_upper = _all_chars[65:65+26]
>>> _ascii_lower = _all_chars[97:97+26]
>>> _just_numbers = _all_chars[48:58]
>>> LOWER_TABLE = "".join(_all_chars[:65] + _ascii_lower + _all_chars[65+26:])
>>> UPPER_TABLE = "".join(_all_chars[:97] + _ascii_upper + _all_chars[97+26:])

np.char.split(s, ' ')
np.char.startswith(s, 'S')
np.char.strip()
np.char.str_len(s)

np.sum(np.char.startswith(s, ' '))  # check for leading spaces
np.sum(np.char.endswith(s0, ' '))   # check for trailing spaces
s0 = np.char.rstrip(s0)

Partitioning:
    lp = np.char.partition(s0, ' ')[:, 0]   # get the left-most partition
    rp = np.char.rpartition(s0, ' ')[:, -1] # get the right-most partition
    lpu, lpcnts= np.unique(lp, return_counts=True)
    rpu, rpcnts= np.unique(rp, return_counts=True)

Queries: d
    np.char.find(c, query) >= 0


References:
-----------

`<https://stackoverflow.com/questions/12983067/how-to-find-unique-vectors-of
-a-2d-array-over-a-particular-axis-in-a-vectorized>`_.

`<https://stackoverflow.com/questions/16970982/find-unique-rows-
in-numpy-array>`_.

`<http://stackoverflow.com/questions/38030054/create-adjacency-matrix-in-
python-for-large-dataset>`_.

np.unique - in the newer version, they use flags to get the sums
:
"""
import sys
import numpy as np
from textwrap import dedent

# ---- others from above , de_punc, _describe, fc_info, fld_info, null_dict,

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]


# ---- decorator and format for structured arrays ----
#
def struct_deco(func):
    """Prints a structured array using `frmt_struct`
    Place this decorator over any function that returns a structured array so
    it can be easily read
    """
    from functools import wraps  # Uncomment, or move it inside the script.
    @wraps(func)
    def wrapper(*args, **kwargs):
        """wrapper function"""
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        argf = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                argf.append("array (shape: {})".format(args[0].shape))
            else:
                argf.append(arg)
        d = {**dict(zip(arg_names, argf)), **kwargs}
        nf = len(max(arg_names, key=len))
        darg = "\n".join(["  {!s:<{}} : {}".format(i, nf, d[i]) for i in d])
        ar = [func.__name__, darg]
        print("\nFunction... {}\nInputs...\n{}\n".format(*ar))
        #
        result = func(*args, **kwargs)   # do the work
        prn_struct(result)              # format the result
        return result                    # for optional use outside.
    return wrapper


def prn_struct(a, edgeitems=3, max_lines=25, wdth=100, decimals=2, prn=True):
    """Format a structured array by reshaping and replacing characters from
    the string representation
    """
    a = np.asanyarray(a)
    nmes = a.dtype.names
    if nmes is not None:
        dtn = "Column names ...\n" + ", ".join(a.dtype.names)
    with np.printoptions(precision=decimals,
                         edgeitems=edgeitems,
                         threshold=max_lines,
                         linewidth=wdth):
        repl = ['[', ']', '(', ')', '"', "'", ',']
        s = str(a.reshape(a.shape[0], 1))
        for i in repl:
            s = np.char.replace(s, i, " ")
        print("{}\n\n{}".format(dtn, s))
     #

# ---- text columns... via char arrays
#
def replace_(a, col=None, from_=None, to_=None):
    """

    """
    #np.char.replace(
    pass

@struct_deco
def find_text(a, col=None, query=None, any_case=True, extract='all'):
    """Query a recarray/structured array for text

    a : recarray/structured array
        Only text columns can be queried
    col : column/field to query
        Only 1 field can be queried at a time for the condition.
    query : string
        The query string
    any_case : boolean
        True, will find records regardless of `case`
    extract: text or list
        `all` extracts all records where the column case is found
        list  extracts the records for only those fields in the list

    >>> find_text(a, col='FULLNAME', query='ABBEY', extract=a.dtype.names[:2])
    """
    #
    # ---- error checking section ----
    e0 = """
    Query error:\n  dtype: {}\n  col: {}\n  query: {}\n  extract: {}"""
    err1 = "\nField not found:\nQuery fields: {}\nArray fields: {}"
    errors = [a.dtype.names is None,
              col is None, query is None, col not in a.dtype.names]
    if sum(errors) > 0:
        print(dedent(e0).format(a.dtype.kind, col, query, extract))
        return None
    if isinstance(extract, (list, tuple)):
        names = a.dtype.names
        r = [i in names for i in extract]
        if sum(r) != len(r):
            print(err1.format(extract, names))
            return None
    #
    # ---- query section
    # convert column values and query to lowercase then query
    c = a[col]
    if any_case:
        c = np.char.lower(c)
        query = query.lower()
    q = np.char.find(c, query) >= 0  # ---- actual query ----
    w = np.where(q)[0]
    if extract == 'all':
        return a[w]
    else:
        return a[list(extract)][w]


#@struct_deco
def tab_count(row, col, r_fld=None, c_fld=None, verbose=False):
    """Crosstabulate 2 fields data arrays, shape (N,), using np.unique.
    scipy.sparse has similar functionality and is faster for large arrays.

    Requires:
    --------
    A 2D array of data with shape(N,) representing two variables.

    row : field/column
        row variable
    col : field/column
        column variable

    Useage:
    ------
    >>> float_min = np.finfo(np.float).min
    >>> float_max = np.finfo(np.float).max
    >>> int_min = np.iinfo(np.int_).min
    >>> int_max = np.iinfo(np.int_).max
    >>> f = r'C:\some\path\your.gdb\your_featureclass'
    >>> null_dict = {'Int_fld': int_min, 'Float_fld': float_min}  # None strings
    >>> flds = ['Int_field', 'Txt01', 'Txt02']  # 2 text fields
    >>> t = arcpy.da.TableToNumPyArray(in_table=f, field_names=flds,
                                      skip_nulls=False)
                                      # , null_value=null_dict) if needed
    >>> rows = t['Txt01']
    >>> cols = t['Txt02']
    >>> ctab, a, result, r, c = crosstab(rows, cols, verbose=False)

    Returns:
    --------
      ctab :
          the crosstabulation result as row, col, count array
      a :
          the crosstabulation in a row, col, count, but filled out whether a
          particular combination exists or not.
      r, c :
          unique values/names for the row and column variables
    """
    def _prn(r, c, r_fld, c_fld, a):
        """fancy print formatting.
        """
        r = r.tolist()
        r.append('Total')
        c = c.tolist()
        c.append('Total')
        r_sze = max([len(str(i)) for i in r]) + 2
        c_sze = [max(len(str(i)), 5) for i in c]
        f_0 = '{{!s:<{}}} '.format(r_sze)
        f_1 = ('{{!s:>{}}} '*len(c)).format(*c_sze)
        frmt = f_0 + f_1
        hdr = 'Row: {}\nCol: {}\n'.format(r_fld, c_fld) + '_' * (r_sze)
        txt = [frmt.format(hdr, *c)]
        txt2 = txt + [frmt.format(r[i], *a[i]) for i in range(len(r))]
        result = "\n".join(txt2)
        return result
    #
    r_fld = [str(r_fld), "Row"][r_fld is None]
    c_fld = [str(c_fld), "Col"][c_fld is None]
    dt = np.dtype([(r_fld, row.dtype), (c_fld, col.dtype)])
    rc = np.asarray(list(zip(row, col)), dtype=dt)
    r = np.unique(row)
    c = np.unique(col)
    u, idx, cnt = np.unique(rc, return_index=True, return_counts=True)
    rcc_dt = u.dtype.descr
    rcc_dt.append(('Count', '<i4'))
    ctab = np.asarray(list(zip(u[r_fld], u[c_fld], cnt)), dtype=rcc_dt)
    c0 = np.zeros((len(r), len(c)), dtype=np.int_)
    rc = [[(np.where(r == i[0])[0]).item(),
           (np.where(c == i[1])[0]).item()] for i in ctab]
    for i in range(len(ctab)):
        rr, cc = rc[i]
        c0[rr, cc] = ctab[i][2]
    tc = np.sum(c0, axis=0)
    c1 = np.vstack((c0, tc))
    tr = np.sum(c1, axis=1)
    counts = np.hstack((c1, tr.reshape(tr.shape[0], 1)))
    if verbose:
        out_tbl = _prn(r, c, r_fld, c_fld, counts)
        print(out_tbl)
    return ctab


#@ struct_deco
def tab_sum(a, r_fld=None, c_fld=None, val_fld=None):
    """Tabular sum of values for two attributes

    Parameters:
    ----------
    a : array
        Structured/recarray
    r_fld, r_col : string
        The fields to be used as the table rows and columns
    val_fld : string
        The field that will be summed for the unique combinations of
        row/column classes

    Returns:
    --------
    A table summarizing the sums for the row/column combinations.
    """
    # ---- Slice the input array using the row/column fields, determine the
    # unique combinations of their attributes.  Create the output dtype
    if a[val_fld].dtype.kind not in ('i', 'f'):
        print("\nThe value field must be numeric")
        return None
    rc = a[[r_fld, c_fld]]
    sum_name = val_fld +'Sum'
    dt = rc.dtype.descr + [(sum_name, '<i4')]
    uniq = np.unique(rc)
    #
    # ----
    out_ = []
    for u in uniq:
        c0, c1 = u
        idx = np.logical_and(a[r_fld]==c0, a[c_fld]==c1)
        val = np.nansum(a[val_fld][idx])
        out_.append([c0, c1, val])
    out_ = np.array(out_)
    z = np.empty((len(out_),), dtype=dt)
    z[r_fld] = out_[:, 0]
    z[c_fld] = out_[:, 1]
    z[sum_name] = out_[:, 2].astype('int32')
    return z



# ---- crosstab from tool, uncomment for testing or tool use
def _demo():
    """Load the sample file for testing
    """
    frmt = """\
    Crosstab results ....
    {}\n
    The array of counts/frequencies....
    {}\n
    Row field:  {}
    Col field:  {}\n
    Row and column headers...
    {}
    {}\n
    And as a fancy output which can be saved to a csv file using
    ....np.savetxt('c:/path/name.csv', array, fmt= '%s', delimiter=', ')\n
    {}
    """
    in_tbl = 'C:/Git_Dan/arraytools/Data/sample_20.npy'  # sample_20.npy
    a = np.load(in_tbl)
    row_fld = 'County'
    col_fld = 'Town'
    rows = a[row_fld]
    cols = a[col_fld]
    ctab = tab_count(rows, cols,
                     r_fld=row_fld,
                     c_fld=col_fld,
                     verbose=True)
    return a, ctab

def _data():
    """base file"""
    in_tbl = 'C:/Git_Dan/arraytools/Data/points_2000.npy'
    a = np.load(in_tbl)
    return a
if __name__ == "__main__":
    """run crosstabulation with data"""
#    ctab, counts, out_tbl = tab_count(a['County'], a['Town'], r_fld='County', c_fld='Town', verbose=False)
#    ctab, a, result, r, c = _demo()

