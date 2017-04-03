# -*- coding: UTF-8 -*-
"""
:Script:   frmts.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-04-02
:Purpose:
:  The frmt_ function is used to provide a side-by-side view of 2,3, and 4D
:  arrays.  Specifically, 3D and 4D arrays are useful and for testing
:  purposes, seeing the dimensions in a different view can facilitate
:  understanding.  For the best effect, the array shapes should be carefully
:  considered. Some guidelines follow.  The middle 'r' part of the shape is
:  not as affected as the combination of the 'd' and 'c' parts.  The array is
:  trimmed beyond the 'wdth' parameter in frmt_.
:
:  Sample the 3D array shape so that the format (d, r, c)
:  is within the 20-21 range for d*c ... for example:
:        integers          floats
:        2, r, 10  = 20    2, r, 8 = 16
:        3, r,  7  = 21    3, 4, 5 = 15
:        4, r,  5  = 20    4, r, 4 = 16
:        5, r,  4  = 20    5, r, 3 = 15
:
:   frmt_(a)  example for a =  np.arange(3*4*7).reshape(3, 4, 7)
:  ---------------------------------------------------
:  : Array... shape (3, 4, 5), ndim 3
:  :
:    0  1  2  3  4    20 21 22 23 24    40 41 42 43 44
:    5  6  7  8  9    25 26 27 28 29    45 46 47 48 49
:   10 11 12 13 14    30 31 32 33 34    50 51 52 53 54
:   15 16 17 18 19    35 36 37 38 39    55 56 57 58 59
:  : sub (0 )        : sub (1 )        : sub (2 )
:
:  The middle part of the shape should also be reasonable should you want
:  to print the results:
:
:  How it works
:
:  a[...,0,:].flatten()
:  array([ 0,  1,  2,  3,  4, 20, 21, 22, 23, 24, 40, 41, 42, 43, 44])
:
:  a[...,0,(0, 1, -2, -1)].flatten()
:  array([ 0,  1,  3,  3, 20, 21, 23, 23, 40, 41, 43, 43])
:
:----------------------------------------------------------------------------
:Functions:  help(<function name>) for help
:=========
: ...public ... private...
:    deline  -  _pre
:    frmt_   - _check, _concat, _row_format
:    frmt_ma - _fix
:    in_by   - _pre_num
: ... see __all__ for a complete listing
:
: (1) ---- col_hdr() ----
:   - produce column headers to align output for formatting purposes
:           1         2         3         4         5         6
:  123456789012345678901234567890123456789012345678901234567890123456789
:  ----------------------------------------------------------------------
:
: (2) ---- deline(a) ----
:     shp = (2,3,4)
:     a = np.arange(np.prod(shp)).reshape(shp)
:     deline(a)
:
:     Main array...
:     ndim: 3 size: 24
:     shape: (2, 3, 4)
:     [[[ 0  1  2  3]
:       [ 4  5  6  7]
:       [ 8  9 10 11]]
:     a[1]....
:      [[12 13 14 15]
:       [16 17 18 19]
:       [20 21 22 23]]]
:
: (3) ---- frmt_(a) ----
:   a = np.arange(2*3*3).reshape(2,3,3)
:   array([[[ 0,  1,  2],
:           [ 3,  4,  5],
:           [ 6,  7,  8]],
:
:          [[ 9, 10, 11],
:           [12, 13, 14],
:           [15, 16, 17]]])
:   f_(a)
:   Array... shape (2, 3, 3), ndim 3, not masked
:    0,  1,  2     9, 10, 11
:    3,  4,  5    12, 13, 14
:    6,  7,  8    15, 16, 17
:   sub (0)       sub (1)
:
: (4) ---- frmt_ma ----
:   :--------------------
:   :Masked array........
:   :  ndim: 2 size: 20
:   :  shape: (5, 4)
:   :
:   :... a[:5, :4] ...
:     -  1  2  3
:     4  5  6  7
:     8  -  -  -
:    12 13 14 15
:    16 17 18  -
:
: (5) frmt_rec(in_array, deci=2, f_names=False, max_rows=-1)
:
: (6) frmt_struct(a, deci=2, f_names=False, prn=False)
:
: (7) ---- in_by ---- indent objects, added automatic support for arrays
:     and optional line numbers
:   - example
:     >>> a = np.arange(2*3*4).reshape(2,3,4)
:     >>> print(art.in_by(a, hdr='---- header ----', nums=True, prefix =".."))
:     ---- header ----
:     00..[[[ 0  1  2  3]
:     01..  [ 4  5  6  7]
:     02..  [ 8  9 10 11]]
:     03..
:     04.. [[12 13 14 15]
:     05..  [16 17 18 19]
:     06..  [20 21 22 23]]]
:
: (8) make_row_format(dim=2, cols=3, a_kind='f', deci=1, a_max=10, a_min=-10)
:
: (9) ---- redent(lines, spaces=4)
:     a = np.arange(3*5).reshape(3,5)
:     >>> print(redent(a))
:     |    [[ 0  1  2  3  4]
:     |     [ 5  6  7  8  9]
:     |     [10 11 12 13 14]]
:
:----------------------------------------------------------------------------
:Notes:
:=====
: ****** column numbering ******
:  d = (('{:<10}')*7).format(*'0123456789'), '0123456789'*7, '-'*70
:  s = "\n{}\n{}\n{}".format(args[0][1:], args[1][1:], args[2]) #*args)
:  print(s)
:           1         2         3         4         5         6
:  123456789012345678901234567890123456789012345678901234567890123456789
:
: ***** Getting default print options, then setting them back *****
:  pr_opt = np.get_printoptions()
:  df_opt = ", ".join(["{}={}".format(i, pr_opt[i]) for i in pr_opt])
:
: ****** Rearranging blocks into columns using np.c_[...] ******
:  a = np.arange(3*2*3).reshape(3,2,3)
:  a_max = a.max()
:  a_min = a.min()
:  aa = np.c_[(a[0],a[1],a[2])]
:  d, r, c = a.shape
:  deci = 1
:  a_kind = a.dtype.kind
:  f = _format_row_test(d, r, c, a_kind, deci, a_max, a_min)
:
:  Row format given
:  d 3, r 2, c 3
:  kind i decimals 1
:  {:3.0f}{:3.0f}{:3.0f}  {:3.0f}{:3.0f}{:3.0f}  {:3.0f}{:3.0f}{:3.0f}
:  0123456789012345678901234567890123456789012345678901234567890123456789
:  0         1         2         3         4         5         6
:
:  r = "\n".join([f.format(*i) for i in aa])
:  print(r)
:  0  1  2    6  7  8   12 13 14
:  3  4  5    9 10 11   15 16 17
:
:  Now change dtype and decimals
:  a_kind = 'f'
:  deci = 2
:  f = _format_row_test(d, r, c, a_kind, deci, a_max, a_min)
: .... snip ....
:  print(r)
:  0.00  1.00  2.00    6.00  7.00  8.00   12.00 13.00 14.00
:  3.00  4.00  5.00    9.00 10.00 11.00   15.00 16.00 17.00
:
: ***** all at once *******
:  a
:  array([[[ 0,  1,  2,  3],
:          [ 4,  5,  6,  7],
:          [ 8,  9, 10, 11]],
:
:         [[12, 13, 14, 15],
:          [16, 17, 18, 19],
:          [20, 21, 22, 23]]])
:  s0,s1,s2 = a.shape
:  b = a.swapaxes(2,1).reshape(s0*s2,s1).T
:  b
:  array([[ 0,  1,  2,  3, 12, 13, 14, 15],
:         [ 4,  5,  6,  7, 16, 17, 18, 19],
:         [ 8,  9, 10, 11, 20, 21, 22, 23]])
:
:Masked array info:
:-----------------
:  a.get_fill_value() # see default_filler dictionary
:  a.set_fill_value(np.NaN)
:  np.ma.maximum_fill_value(a)   -inf
:  np.ma.minimum_fill_value(a)    inf
:  default_filler =
:     {'b': True, 'c': 1.e20 + 0.0j, 'f': 1.e20, 'i': 999999,'O': '?',
:      'S': b'N/A', 'u': 999999,'V': '???','U': sixu('N/A')}
:
:References:
:----------
: b.transpose(1,2,0)[:,:,::-1] *** tip *** reorder from after transpose
:  or even a swapaxes the ::-1 does the reversing... same as [...,::-1]
:----------
"""

# ---- imports, formats, constants ----

import sys
import numpy as np
from textwrap import dedent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter=ft)
pr_opt = np.get_printoptions()
df_opt = ", ".join(["{}={}".format(i, pr_opt[i]) for i in pr_opt])

script = sys.argv[0]

__all__ = ['col_hdr',
           'deline',
           'frmt_',
           'frmt_ma',
           'frmt_rec',
           'frmt_struct',
           'in_by',
           'make_row_format',
           'redent',
           '_demo',
           '_ma_demo']


# ----------------------------------------------------------------------
# (1) col_hdr ... code section .....
def col_hdr():
    """Print numbers from 1 to 70 to show column positions"""
    args = [(('{:<10}')*7).format(*'0123456789'), '0123456789'*7, '-'*70]
    s = "\n{}\n{}\n{}".format(args[0][1:], args[1][1:], args[2])  # *args)
    print(s)


# ----------------------------------------------------------------------
# (2) deline ... code section .....
def deline(a, header="", prefix="  ."):
    """Remove extraneous lines from array output
    :  More useful for long arrays with ndim >= 3
    :Requires:
    :--------
    : a - anything that can be put into array form
    : header - an optional header
    : prefix - could be just spaces or something like shown
    :Returns:
    :-------
    :  a string for printing
    """
    if not isinstance(a, (list, tuple, np.ndarray)):
        return "list, tuple or ndarray required"
    a = np.asanyarray(a)
    header += "\nMain array... \nshape: {}".format(a.shape)
    f1 = "[{},...] {}"
    out = [header]
    c = 0

    def _pre(obj):
        for line in obj.splitlines(False):
            frmt = "{}{}".format(prefix, line)
            yield frmt
    for i in a:
        a_s = f1.format(c, i.shape)
        out.append(a_s)
        out.extend(_pre(str(i)))
        c += 1
    f = "\n".join([i for i in out if i != prefix])
    return f


# ----------------------------------------------------------------------
# (3) frmt_ .... code section
def frmt_(a, deci=4, wdth=100, title="Array", prefix="  .", prn=True):
    """Format number arrays by row, and print
    :Requires:
    :--------
    : a - an array of int or float dtypes, 1, 2, 3 and 4D arrays tested.
    : deci - decimal places for floating point numbers
    : wdth - Default width for onscreen and printing, output beyond this
    :   length will be truncated with a warning.  Reshape to overcome.
    : title - The default title, change to provide more information.
    :Returns:
    :--------
    : prints the array with the 1st dimension flattened-like by row
    :Notes:
    : w_frmt  width formatter
    : m_frmt  max number formatter to get max. number of characters
    """
    def _check(a):
        """ check dtype and max value for formatting information"""
        return a.shape, a.ndim, a.dtype.kind, a.max(), a.min()
    #

    def _concat(rows, r_fmt, wdth, prefix):
        """print the subset to maximimum width"""
        end = ["", "...."][len(r_fmt.format(*rows[0])) > wdth]
        txt = prefix
        rw = [r_fmt.format(*v)[:wdth] + end for v in rows]
        txt += ("\n" + prefix).join(rw) + "\n"
        return txt
    #

    def _row_format(d, r, c, a_kind, deci, a_min, a_max):
        """Format the row based on input parameters
        : d, r, c =a.shape[:-3]  last 3 dimensions of array shape
        : a_kind - a.dtype.kind  array kind ie integer or float
        """
        if a_kind == 'f':
            w_, m_ = [':{}.{}f', '{:0.{}f}']
        else:
            w_, m_ = [':{}.0f', '{:0.0f}']
        m = max(len(m_.format(a_max, deci)), len(m_.format(a_min, deci))) + 1
        w_fmt = w_.format(m, deci)
        r_fmt = (('{' + w_fmt + '}') * c + '  ') * d
        return r_fmt
    #
    # ---- begin constructing the array format ----
    txt = ""
    a = np.asanyarray(a)
    if a.ndim < 3:
        return "Array is not 3D or 4D"
    fv = ""
    if np.ma.isMaskedArray(a):
        fv = ", masked array, fill value {}".format(a.get_fill_value())
        a = a.data
    # ---- run _check ----
    a_shp, a_dim, a_kind, a_min, a_max = _check(a)
    #
    # ---- correct dtype, get formats ----
    if (a_kind in ('i', 'f')) and (a_dim >= 3):
        args = title, a_shp, a_dim, fv
        txt = "{}...\n-shape {}, ndim {}{}".format(*args)
        d, r, c = a_shp[-3:]
        row_frmt = _row_format(d, r, c, a_kind, deci, a_min, a_max)
        if (a_dim == 3):
            rows = [a[..., i, :].flatten() for i in range(r)]
            txt += "\n" + _concat(rows, row_frmt, wdth, prefix)
        else:
            d4, d, r, c = a_shp
            fm = "\n--- array[{}] => ({}, {}, {})"
            for d3 in range(d4):
                txt += fm.format(d3, d, r, c) + "\n"
                a_s = a[d3]
                rows = [a_s[..., i, :].flatten() for i in range(r)]
                txt += _concat(rows, row_frmt, wdth, prefix)
    else:
        txt = "Only integer and float arrays with ndim >= 3 supported"
    if prn:
        print(txt)
    else:
        return txt


# ----------------------------------------------------------------------
# (4) frmt_ma .... code section
def frmt_ma(a, prn=True, prefix="  ."):
    """Format a masked array to preserve columns widths and style.
    :Requires
    :--------
    :  a - masked array
    :  prn - True to print
    :  prefix - can be "" for no indentation or "   " or the default
    :Returns
    :-------
    :  Returns a print version of a masked array formatted with masked
    :  values and appropriate spacing.
    :  b = a.reshape(2,4,5) for 3d
    :Notes
    :-----
    :  Get a string representation of the array.  Determine the maximum value
    :  and format each column using that value.  Pad the result with a leader
    :  or replace the prefix with ''
    """
    def _fix(v, tmp, prefix):
        """ sub array adjust"""
        r = [['[[', " "], ['[', ""], [']', ""], [']]', ""]]
        for i in r:
            tmp = tmp.replace(i[0], i[1])
        tmp0 = [i.strip().split(' ') for i in tmp.split('\n')]
        N = len(tmp0[0])
        out = [""]
        for i in range(len(tmp0)):
            out.append((ft*N).format(*tmp0[i]))
        jn = "\n" + prefix
        v += jn.join([i for i in out])
        v += '\n'
        return v
    # ---- main section ----
    dim = a.ndim
    shp = a.shape
    a_max = len(str(np.ma.max(a)))
    ft = '{:>' + str(a_max + 1) + '}'
    v = "\n:Masked array... ndim: {}\n".format(dim)
    if dim == 2:
        v += "\n:.. a[:{}, :{}] ...".format(*shp)
        v = _fix(v, str(a), prefix)
    elif dim == 3:
        for d0 in range(shp[0]):  # dimension blocks
            v += "\n:.. a[{}, :{}, :{}] ...".format(d0, *a[d0].shape)
            v = _fix(v, str(a[d0]), prefix)
    if prn:
        print(v)
    else:
        return v


# ----------------------------------------------------------------------
# (5) frmt_rec .... code section
def frmt_rec(in_array, deci=2, f_names=False, max_rows=-1):
    """Format recarray/structured array with column names and row numbers.
    :  Checks the first 250 rows to determine row format
    :Requires:
    :--------
    :  f_names  - boolean, use existing field names or alphabetic headers
    :  max_rows - -1, is for all, otherwise numeric.
    :  prn      - boolean, print while running, always returns output
    """
    rows = min(in_array.shape[0], 250)
    a = in_array[:rows]
    cols = len(a[0])
    title = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ", a.dtype.names][f_names]
    f_wdth = [len(i) for i in title]
    c_wdth = [max([len(str(item)) + 1 for item in col]) for col in zip(*a)]
    widths = [max(i) for i in zip(f_wdth, c_wdth)]
    f = [" {{!s: >{}}} ".format(width + 1) for width in widths]
    c = [a[0][i].dtype.kind for i in range(len(a[0]))]
    xx = ["a", "s", "S", "S1", "U", "V"]
    txt = [[f[i], f[i].replace("!s: >", "!s: <")][c[i] in xx]
           for i in range(cols)]
    frmt = " {:03.0f} " + "".join([i for i in txt])
    m = "--n--" + "".join(["  {{!s:^{}}} ".format(width) for width in widths])
    hdr = m.format(*title) + "\n"
    msg = hdr + "-"*len(hdr) + "\n"
    idx = 0
    # ---- begin msg construction ----
    for row in in_array:
        msg += frmt.format(idx, *row) + "\n"
        idx += 1
    return msg


# ----------------------------------------------------------------------
# (6) frmt_struct .... code section
#
def _col_format(a, deci=0):
    """Determine column format given a desired number of decimal places.
    :  Used by frmt_struct
    """
    a_kind = a.dtype.kind
    if a_kind in ('f', 'c'):
        w_, m_ = [':> {}.{}f', '{:> 0.{}f}']
    elif a_kind in ('i', 'u'):
        w_, m_ = [':> {}.0f', '{:> 0.0f}']
        deci = 0
    if a_kind in ('f', 'i'):
        a_max, a_min = np.round(np.sort(a[[0, -1]]), deci)
        col_wdth = max(len(m_.format(a_max, deci)),
                       len(m_.format(a_min, deci))) + 1  # m + dec if needed
        c_fmt = w_.format(col_wdth, deci)
    else:
        col_wdth = max([len(i) for i in a])
        c_fmt = "!s:>" + "{}".format(col_wdth)
    return c_fmt, col_wdth


def frmt_struct(a, deci=2, f_names=False, prn=False):
    """Format a structured array with a mixed dtype.
    :Requires
    :-------
    : a - a structured/recarray
    : deci - to facilitate printing, this value is the number of decimal
    :        points to use for all floating point fields.
    : _col_format - does the actual work of obtaining a representation of
    :  the column format.
    :Notes
    :-----
    :  It is not really possible to deconstruct the exact number of decimals
    :  to use for float values, so a decision had to be made to simplify.
    """
    nms = a.dtype.names
    N = len(nms)
    title = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:N], nms][f_names]
    # ---- get the column formats from ... _col_format ----
    dts = []
    wdths = []
    for i in nms:
        c_fmt, col_wdth = _col_format(a[i], deci)
        dts.append(c_fmt)
        wdths.append(col_wdth)
    rf = " ".join([('{' + i + '}') for i in dts])
    hdr = ["!s:>" + "{}".format(wdths[i]) for i in range(N)]
    hdr2 = " ".join(["{" + hdr[i] + "}" for i in range(N)])
    header = hdr2.format(*title)
    txt = [header]
    for i in range(a.shape[0]):
        row = rf.format(*a[i])
        txt.append(row)
    if prn:
        for i in txt:
            print(i)
    msg = "\n".join([i for i in txt])
    return msg


# ----------------------------------------------------------------------
# (7) in_by .... code section
def in_by(obj, hdr="", nums=False, prefix="  "):
    """textwrap.indent variant for python 2.7 or a substitute for
    :  any version of python.  The function stands for 'indent by'.
    :Requires:
    :--------
    :  obj - obj to indent, List, tuple, ndarray converted to strings
    :    first. You can use repr representation before using if needed.
    :  hdr - optional header
    :  nums - boolean, add line numbers
    :  prefix - text to use for indent ie '  ' for 2 spaces or '....'
    :Reference:
    :---------
    :  https://docs.python.org/3.7/library/textwrap.html for python >3.3
    :Notes:
    :-----
    :  Header and line numbers options added.
    """
    if hdr != "":
        hdr = "{}\n".format(hdr)
    if isinstance(obj, (list, tuple, np.ndarray)):
        obj = str(obj)

    def _pre_num():
        c = 0
        for line in obj.splitlines(True):
            if nums:
                frmt = "{:>02}{}{}".format(c, prefix, line)
            else:
                frmt = "{}{}".format(prefix, line)
            yield frmt
            c += 1
    out = hdr + "".join(_pre_num())
    return out


# ----------------------------------------------------------------------
# (8) make_row_format ... code section .....
def make_row_format(dim=2, cols=3, a_kind='f', deci=1, a_max=10, a_min=-10):
    """Format the row based on input parameters
    : dim - number of dimensions
    : cols - columns per dimension
    : a_kind, deci, a_max and a_min allow you to specify a data type, number
    :   of decimals and maximum and minimum values to test formatting.
    """
    if a_kind not in ['f', 'i']:
        a_kind = 'f'
    w_, m_ = [[':{}.0f', '{:0.0f}'], [':{}.{}f', '{:0.{}f}']][a_kind == 'f']
    m_fmt = max(len(m_.format(a_max, deci)), len(m_.format(a_min, deci))) + 1
    w_fmt = w_.format(m_fmt, deci)
    s = (m_fmt + 1)*cols + 1
    hdr = (('sub ({:<1.0f})' + " "*s)[:s+6])*dim
    row_frmt = ((('{' + w_fmt + '}')*cols + '  ')*dim).strip()
    frmt = "Row format: dim cols: ({}, {})  kind: {} decimals: {}\n\n{}"
    print(dedent(frmt).format(dim, cols, a_kind, deci, row_frmt))
    a = np.random.randint(a_min, a_max+1, dim*cols)  # [1, 5, 10, -1, 5, -10]))
    col_hdr()  # run col_hdr to produce the column headers
    print(row_frmt.format(*a))
    return row_frmt


# ----------------------------------------------------------------------
# (9) redent .... code section
def redent(lines, spaces=4):
    """Strip and reindent by num_spaces, a sequence of lines
    :  lines - text or what can be made text
    :  Use str() or repr() on the inputs if you want control on form
    : - see in_by for more options
    """
    lines = str(lines).splitlines()
    sp = [len(ln) - len(ln.lstrip()) for ln in lines]
    spn = " "*spaces
    out = list(zip(lines, sp))
    ret = "\n".join(["{0}{1!s:>{2}}".format(spn, *ln) for ln in out])
    return ret


# ----------------------------------------------------------------------
# ---- demos ----

def _demo():
    """ small samples """
    sh = [2, 2, 4, 2]
    fac = 1
    a = np.arange(np.prod(sh)).reshape(*sh)*fac
    print(deline(a, header="deline demo...", prefix="  ."))
    print("\nf1_ demo....")
    frmt_(a, deci=4, wdth=100, title="Array a...", prn=True)
    a1 = [np.array([1, 2, 3]), np.arange(8).reshape(4, 2),
          np.arange(3*3*2).reshape(3, 3, 2),
          np.arange(2*2*4*2).reshape(2, 2, 4, 2)]
    print(deline(a1, header="deline with object array...", prefix="   -"))
    return a


def _ma_demo():
    """Produce a simple masked array and format it using frmt_ma
    :  Change the values to suit
    """
    np.ma.masked_print_option.set_display('-')
    a = np.array([[100, 1, 2, -99, 99], [5, 6, 7, -99, 9],
                  [-99, 11, -99, 13, -99], [15, 16, 17, 18, 19],
                  [-99, -99, 22, 23, 24], [25, 26, 27, 28, -99],
                  [30., 31, 32, 33, -99], [35, -99, 37, 38, 39]],
                 dtype='<f8')
    m = np.where(a == -99, 1, 0)
    mask_val = -99
    a = np.ma.array(a, mask=m, fill_value=mask_val)
    # ---- test output ----
    print("Sample run of frmt_ma...")
    frmt_ma(a, prn=True)
    print("\nArray reshaped two 3D")
    b = a.reshape(2, 4, 5)
    frmt_ma(b, prn=True)
    # return a, b


def _struct_demo():
    """load and print a structured array
    """
    pth = _struct_demo.__code__.co_filename
    pth = pth.split("\\")[:-1] + ["sample_data.npy"]
    aa = np.load("/".join(pth))
    a = aa[['OID', 'NEAR_FID', 'FROM_X', 'FROM_Y']]
    print(frmt_struct(a[:3], deci=2, f_names=True, prn=False))
    print(frmt_struct(a[:3], deci=2, f_names=False, prn=False))
    print(frmt_rec(a[:3], deci=2, f_names=False))
    print(frmt_rec(a[:3], deci=2, f_names=True))
    # return a


# -------------------------
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
#    row_frmt = make_row_format()
#    a = _demo()
#    a, b = _ma_demo()
#    a = _struct_demo()
