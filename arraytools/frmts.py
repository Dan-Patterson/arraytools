# -*- coding: UTF-8 -*-
"""
frmts.py  formatting arrays
===========================

Script:   frmts.py

Author:   Dan_Patterson@carleton.ca

Modified: 2018-03-28

Purpose:

  The frmt_ function is used to provide a side-by-side view of 2, 3, and 4D
  arrays.  Specifically, 3D and 4D arrays are useful and for testing
  purposes, seeing the dimensions in a different view can facilitate
  understanding.  For the best effect, the array shapes should be carefully
  considered. Some guidelines follow.  The middle 'r' part of the shape is
  not as affected as the combination of the 'd' and 'c' parts.  The array is
  trimmed beyond the 'wdth' parameter in frmt_.

  Sample the 3D array shape so that the format (d, r, c)
  is within the 20-21 range for d*c ... for example::
        integers          floats
        2, r, 10  = 20    2, r, 8 = 16
        3, r,  7  = 21    3, 4, 5 = 15
        4, r,  5  = 20    4, r, 4 = 16
        5, r,  4  = 20    5, r, 3 = 15


  >>> frmt_(a)  example for a =  np.arange(3*4*5).reshape(3, 4, 5)
  ---------------------------------------------------
  :Array...
  :-shape (3, 4, 5), ndim 3
  :  .  0  1  2  3  4    20 21 22 23 24    40 41 42 43 44
  :  .  5  6  7  8  9    25 26 27 28 29    45 46 47 48 49
  :  . 10 11 12 13 14    30 31 32 33 34    50 51 52 53 54
  :  . 15 16 17 18 19    35 36 37 38 39    55 56 57 58 59
  :  .   sub (0 )        : sub (1 )        : sub (2 )

  The middle part of the shape should also be reasonable should you want
  to print the results:

  How it works

  >>> a[...,0,:].flatten()
  array([ 0,  1,  2,  3,  4, 20, 21, 22, 23, 24, 40, 41, 42, 43, 44])

  >>> a[...,0,(0, 1, -2, -1)].flatten()
  array([ 0,  1,  3,  3, 20, 21, 23, 23, 40, 41, 43, 43])


Functions:
=========
help(<function name>) for help

::

    public  -  private...
    deline  -  _pre
    frmt_   - _check, _concat, _row_format
    frmt_ma - _fix
    in_by   - _pre_num

 ... see __all__ for a complete listing

1(a) col_hdr() :

produce column headers to align output for formatting purposes

``.........1.........2.........3.........4.........5.........6.........
123456789012345678901234567890123456789012345678901234567890123456789``

----------------------------------------------------------------------


1(b)  deline(a)::

     shp = (2,3,4)
     a = np.arange(np.prod(shp)).reshape(shp)
     deline(a)

     Main array...
     ndim: 3 size: 24
     shape: (2, 3, 4)
     [[[ 0  1  2  3]
       [ 4  5  6  7]
       [ 8  9 10 11]]
     a[1]....
      [[12 13 14 15]
       [16 17 18 19]
       [20 21 22 23]]]

(1c) in_by

indent objects, added automatic support for arrays and optional line numbers
::
     a = np.arange(2*3*4).reshape(2,3,4)
     print(art.in_by(a, hdr='---- header ----', nums=True, prefix =".."))
     ---- header ----
     00..[[[ 0  1  2  3]
     01..  [ 4  5  6  7]
     02..  [ 8  9 10 11]]
     03..
     04.. [[12 13 14 15]
     05..  [16 17 18 19]
     06..  [20 21 22 23]]]

(1d)  redent(lines, spaces=4)
::
     a = np.arange(3*5).reshape(3,5)
     >>> print(redent(a))
     |    [[ 0  1  2  3  4]
     |     [ 5  6  7  8  9]
     |     [10 11 12 13 14]]

(2) frmt_(a)
::
   a = np.arange(2*3*3).reshape(2,3,3)
   array([[[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8]],

          [[ 9, 10, 11],
           [12, 13, 14],
           [15, 16, 17]]])
   frmt_(a)
   Array... shape (2, 3, 3), ndim 3, not masked
    0,  1,  2     9, 10, 11
    3,  4,  5    12, 13, 14
    6,  7,  8    15, 16, 17
   sub (0)       sub (1)

(3) frmt_ma
::
    :--------------------
    :Masked array........
    :  ndim: 2 size: 20
    :  shape: (5, 4)
    :
    :... a[:5, :4] ...
      -  1  2  3
      4  5  6  7
      8  -  -  -
     12 13 14 15
     16 17 18  -


(4) frmt_rec(in_array, deci=2, use_names=False, max_rows=-1)
::
    Format ... C:/Git_Dan/arraytools/Data/sample_20.npy
    record/structured array, with and without field names.
    --n-- OBJECTID   f0  County  Town  Facility  Time``
    -------------------------------------------------
    000         1    0       B    A_      Hall    26
    001         2    1       C    C_      Hall    60
    002         3    2       D    A_      Hall    42

    --n-- C00  C01  C02  C03  C04   C05
    -----------------------------------
    000    1    0    B   A_ Hall    26
    001    2    1    C   C_ Hall    60
    002    3    2    D   A_ Hall    42


(4a) make_row_format
::
    make_row_format(dim=2, cols=3, a_kind='f', deci=1,
                    a_max=10, a_min=-10, prn=False)
    '{:6.1f}{:6.1f}{:6.1f}  {:6.1f}{:6.1f}{:6.1f}'

(5) form_
::
  form_(a, deci=2, wdth=100, title="Array", prefix=". . ", prn=True)

  Array... ndim: 3  shape: (2, 3, 3)
  . .   0  1  2    9 10 11
  . .   3  4  5   12 13 14
  . .   6  7  8   15 16 17


Notes:
=====

**column numbering**

>>> d = (('{:<10}')*7).format(*'0123456789'), '0123456789'*7, '-'*70
>>> s = '\n{}\n{}\n{}'.format(args[0][1:], args[1][1:], args[2]) #*args)
>>> print(s)
             1         2         3         4         5         6
    123456789012345678901234567890123456789012345678901234567890123456789


**Getting default print options, then setting them back **

>>> pr_opt = np.get_printoptions()
>>> df_opt = ", ".join(["{}={}".format(i, pr_opt[i]) for i in pr_opt])


** Rearranging blocks into columns using np.c_[...] **

>>>  a = np.arange(3*2*3).reshape(3, 2, 3)
>>>  a_max = a.max()
>>>  a_min = a.min()
>>>  aa = np.c_[(a[0], a[1], a[2])]
>>>  d, r, c = a.shape
>>>  deci = 1
>>>  a_kind = a.dtype.kind
>>>  f = _format_row_test(d, r, c, a_kind, deci, a_max, a_min)

::
    Row format given
    d 3, r 2, c 3
    kind i decimals 1
    {:3.0f}{:3.0f}{:3.0f}  {:3.0f}{:3.0f}{:3.0f}  {:3.0f}{:3.0f}{:3.0f}
    0123456789012345678901234567890123456789012345678901234567890123456789
    0         1         2         3         4         5         6

>>>  r = `\\n`.join([f.format(*i) for i in aa])
>>>  print(r)
  0  1  2    6  7  8   12 13 14
  3  4  5    9 10 11   15 16 17

>>> # Now change dtype and decimals
>>>  a_kind = 'f'
>>>  deci = 2
>>>  f = _format_row_test(d, r, c, a_kind, deci, a_max, a_min)
 .... snip ....
>>>  print(r)
  0.00  1.00  2.00    6.00  7.00  8.00   12.00 13.00 14.00
  3.00  4.00  5.00    9.00 10.00 11.00   15.00 16.00 17.00


** all at once **

>>> a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])

>>>  s0, s1, s2 = a.shape
>>>  b = a.swapaxes(2, 1).reshape(s0*s2, s1).T
>>>  b
  array([[ 0,  1,  2,  3, 12, 13, 14, 15],
         [ 4,  5,  6,  7, 16, 17, 18, 19],
         [ 8,  9, 10, 11, 20, 21, 22, 23]])


Masked array info:
------------------

>>>  a.get_fill_value() # see default_filler dictionary
>>>  a.set_fill_value(np.NaN)
>>>  np.ma.maximum_fill_value(a)   -inf
>>>  np.ma.minimum_fill_value(a)    inf
>>>  default_filler =
     {'b': True, 'c': 1.e20 + 0.0j, 'f': 1.e20, 'i': 999999,'O': '?',
      'S': b'N/A', 'u': 999999,'V': '???','U': sixu('N/A')}


References:
-----------

>>> b.transpose(1, 2, 0)[:,:,::-1]
>>> # ** tip *** reorder from after transpose or even a swapaxes
>>> # the ::-1 does the reversing... same as [...,::-1]


----------
"""

# ---- imports, formats, constants ----

import sys
import numpy as np
from textwrap import dedent, indent

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{!r: 0.3f}'.format}
edge = 3
ln_wdth = 100
np.set_printoptions(edgeitems=edge, linewidth=ln_wdth, precision=3,
                    suppress=True, nanstr='-n-', threshold=60, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

pr_opt = np.get_printoptions()
df_opt = ", ".join(["{}={}".format(i, pr_opt[i]) for i in pr_opt])

script = sys.argv[0]

__all__ = ['col_hdr',
           'deline',
           'in_by',
           'redent',
           '_chunks',
           'frmt_',
           'frmt_ma',
           'frmt_rec', 'pd_',
           'make_row_format',
           'form_',
           ]


# ----------------------------------------------------------------------
# (1) Short, or reused code section
#
# (1a) col_hdr ... code section .....
def col_hdr():
    """Print numbers from 1 to 70 to show column positions"""
    args = [(('{:<10}')*7).format(*'0123456789'), '0123456789'*7, '-'*70]
    s = "\n{}\n{}\n{}".format(args[0][1:], args[1][1:], args[2])  # *args)
    print(s)


# ----------------------------------------------------------------------
# (1b) deline ... code section .....
def deline(a, header="", prefix="  .", prn=True):
    """Remove extraneous lines from array output.
    More useful for long arrays with ndim >= 3

    Requires:
    --------
    `a` : anything
        anything that can be put into array form
    `header` :
        an optional header
    `prefix` : text
        could be just spaces or something like shown

    Returns:
    -------
        A string for printing
    """
    if not isinstance(a, (list, tuple, np.ndarray)):
        return "list, tuple or ndarray required"
    a = np.asanyarray(a)
    shp0 = a.shape[-1]
    ln_wdth = pr_opt['linewidth']
    if shp0 <= ln_wdth:
        np.set_printoptions(edgeitems=ln_wdth//2)
    header += "\nMain array... \nshape: {}".format(a.shape)
    f1 = (":arr[{}" + ", :{}"*len(a.shape[1:]) + "]")
    out = [header]
    c = 0

    def _pre(obj):
        for line in obj.splitlines(False):
            frmt = "{}{}".format(prefix, line)
            yield frmt
    for i in a:
        a_s = f1.format(c, *i.shape)  # ---- uses f1 format above
        out.append(a_s)
        out.extend(_pre(str(i)))
        c += 1

    f = "\n".join([i for i in out if i != prefix])
    if prn:
        print(f)
        np.set_printoptions(edgeitems=edge, linewidth=ln_wdth)
    else:
        np.set_printoptions(edgeitems=edge, linewidth=ln_wdth)
        return f


# ---------------------------------------------------------------------------
# (1c) in_by .... code section
def in_by(obj, hdr="", nums=False, prefix="   .", prn=True):
    """A `textwrap.indent` variant for python 2.7 or a substitute for
    any version of python.  The function stands for `indent by`.

    Requires:
    --------
    `obj` : object that can be cast as string
        obj to indent, List, tuple, ndarray converted to strings
        first. You can use repr representation before using if needed.
    `hdr` : text
        optional header
    `nums` : boolean
        True to add line numbers
    `prefix` : test
        Text to use for indent ie '  ' for 2 spaces or '....'

    Reference:
    ---------
    [1] https://docs.python.org/3.7/library/textwrap.html for python > 3.3

    Notes:
    -----
        Header and line numbers options added.
    """
    if hdr != "":
        hdr = "\n{}\n".format(hdr)
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
    if prn:
        print(out)
    else:
        return out


# ----------------------------------------------------------------------
# (1d) redent .... code section
def redent(lines, spaces=4):
    """Strip and reindent by num_spaces, a sequence of lines
    `lines` : text
        Text or what can be made text
        Use str() or repr() on the inputs if you want control on form

    See also:
    --------
        See `in_by` for more options
    """
    lines = str(lines).splitlines()
    sp = [len(ln) - len(ln.lstrip()) for ln in lines]
    spn = " "*spaces
    out = list(zip(lines, sp))
    ret = "\n".join(["{0}{1!s:>{2}}".format(spn, *ln) for ln in out])
    return ret


# ----------------------------------------------------------------------
# (1e) _chunks .... code section
def _chunks(s, n):
    """Produce n-character chunks from s."""
    for start in range(0, len(s), n):
        yield s[start:start+n]


# ----------------------------------------------------------------------
# (2) frmt_ .... code section
def frmt_(a, deci=2, wdth=100, title="Array", prefix="  .", prn=True):
    """Format number arrays by row, and print

    Requires:
    --------
    `a` : array
        An array of int or float dtypes, 1, 2, 3 and 4D arrays tested.
    `deci` - int
        Decimal places for floating point numbers
    `wdth` : int
        Default width for onscreen and printing, output beyond this
        length will be truncated with a warning.  Reshape to overcome.
    `title` : text
        The default title, change to provide more information.

    Returns:
    --------
        Prints the array with the 1st dimension flattened-like by row

    Notes:
    -----
    `w_frmt` :  width formatter

    `m_frmt` :  max number formatter to get max. number of characters
    """
    def _check(a):
        """ check dtype and max value for formatting information"""
        return a.shape, a.ndim, a.dtype.kind, a.max(), a.min()
    # ----

    def _concat(rows, r_fmt, wdth, prefix):
        """print the subset to maximimum width"""
        end = ["", "...."][len(r_fmt.format(*rows[0])) > wdth]
        txt = prefix
        rw = [r_fmt.format(*v)[:wdth] + end for v in rows]
        txt += ("\n" + prefix).join(rw)  # + "\n"
        return txt
    # ----

    def _row_format(d, r, c, a_kind, deci, a_min, a_max):
        """Format the row based on input parameters
        - d, r, c = a.shape[:-3]  last 3 dimensions of array shape
        - a_kind - a.dtype.kind  array kind ie integer or float
        """
        if a_kind == 'f':
            w_, m_ = [':{}.{}f', '{:0.{}f}']
        else:
            w_, m_ = [':{}.0f', '{:0.0f}']
        m = max(len(m_.format(a_max, deci)), len(m_.format(a_min, deci))) + 1
        w_fmt = w_.format(m, deci)
        r_fmt = (('{' + w_fmt + '}') * c + '  ') * d
        return r_fmt

    def d4_frmt(a_shp, a, txt, a_dim):
        """Dealing with 4, 5 ?D arrays"""
        d4, d, r, c = a_shp
        hdr = "\n" + "-"*25
        fm = hdr + "\n-({}, + ({}, {}, {})"
        if a_dim == 5:
            fm = "\n--(.., {}, + ({}, {}, {})"
        t = ""
        for d3 in range(d4):
            t += fm.format(d3, d, r, c) + "\n"
            a_s = a[d3]
            rows = [a_s[..., i, :].flatten() for i in range(r)]
            t += _concat(rows, row_frmt, wdth, prefix)
        return t
    #
    # ---- begin constructing the array format ----
    txt = ""
    a = np.asanyarray(a)
    # ---- run _check ----
    a_shp, a_dim, a_kind, a_min, a_max = _check(a)
    if a_dim < 3:
        if a_dim == 2:
            a = a.reshape((1,) + a.shape)
            a_dim = a.ndim
            a_shp = a.shape
        else:
            return "Array is not >= 2D"
    fv = ""
    if np.ma.isMaskedArray(a):
        fv = ", masked array, fill value {}".format(a.get_fill_value())
        a = a.data
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
        elif (a_dim == 4):
            d4, d, r, c = a_shp
            t = d4_frmt(a_shp, a, txt, a_dim)
            txt += t
        elif (a_dim == 5):
            d5, d4, d, r, c = a_shp
            hdr = "\n" + "-"*25
            for i in range(d5):
                txt += hdr + '\n--({}, ..'.format(i)
                t = d4_frmt(a_shp[1:], a[i], txt, a_dim)
                txt += t
    else:
        txt = "Only integer and float arrays with ndim >= 2 supported"
    if prn:
        print(txt)
    else:
        return txt


# ----------------------------------------------------------------------
# (3) frmt_ma .... code section
def frmt_ma(a, prn=True, prefix="  ."):
    """Format a masked array to preserve columns widths and style.

    Requires
    --------
    `a` : masked array
        A masked array
    `prn` : Boolean
        True to print
    `prefix` : text
        Can be "" for no indentation or "   " or the default

    Returns
    -------
        Returns a print version of a masked array formatted with masked
        values and appropriate spacing.
        b = a.reshape(2,4,5) for 3d

    Notes
    -----
        Get a string representation of the array.  Determine the maximum value
        length of a string of the values in the array  and format each column
        using that value.  Pad the result with a leader or replace the prefix
        with ''
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
            N = len(tmp0[i])
            out.append((frmt*N).format(*tmp0[i]))
        jn = "\n" + prefix
        v += jn.join([i for i in out])
        v += '\n'
        return v
    # ---- main section ----
    np.set_printoptions(threshold=1000)
    dim = a.ndim
    shp = a.shape
    a_max = max(len(str(np.ma.max(a))), len(str(np.ma.min(a))))  # largest str
    frmt = '{:>' + str(a_max + 1) + '}'
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
# (4) frmt_rec .... code section
#  frmt_rec requires _col_format
def _col_format(a, c_name="c00", deci=0):
    """Determine column format given a desired number of decimal places.
    Used by frmt_struct.

    `a` : column
        A column in an array.
    `c_name` : text
        column name
    `deci` : int
        Desired number of decimal points if the data are numeric

    Notes:
    -----
        The field is examined to determine whether it is a simple integer, a
        float type or a list, array or string.  The maximum width is determined
        based on this type.

        Checks were also added for (N,) shaped structured arrays being
        reformatted to (N, 1) shape which sometimes occurs to facilitate array
        viewing.  A kludge at best, but it works for now.
    """
    a_kind = a.dtype.kind
    if a_kind in ('i', 'u'):  # ---- integer type
        w_, m_ = [':> {}.0f', '{:> 0.0f}']
        col_wdth = len(m_.format(a.max())) + 1
        col_wdth = max(len(c_name), col_wdth) + 1  # + deci
        c_fmt = w_.format(col_wdth, 0)
    elif a_kind == 'f' and np.isscalar(a[0]):  # ---- float type with rounding
        w_, m_ = [':> {}.{}f', '{:> 0.{}f}']
        a_max, a_min = np.round(np.sort(a[[0, -1]]), deci)
        col_wdth = max(len(m_.format(a_max, deci)),
                       len(m_.format(a_min, deci))) + 1
        col_wdth = max(len(c_name), col_wdth) + 1
        c_fmt = w_.format(col_wdth, deci)
    # ---- lists, arrays, strings. Check for (N,) vs (N,1)
    # I made some changes in how col_wdth is determined, old is commented
    else:
        if a.ndim == 1:  # ---- check for (N, 1) format of structured array
            a = a[0]
        dt = a.dtype.descr[0][1]
        col_wdth = int("".join([i for i in dt if i.isdigit()]))
#       col_wdth = max([len(str(i)) for i in a])
        col_wdth = max(len(c_name), col_wdth) + 1  # + deci
        c_fmt = "!s:>" + "{}".format(col_wdth)
    return c_fmt, col_wdth


def pd_(a, deci=2, use_names=True, prn=True):
    """see help for `frmt_rec`..."""
    ret = frmt_rec(a, deci=deci, use_names=use_names, prn=prn)
    return ret


def frmt_rec(a, deci=2, use_names=True, prn=True):
    """Format a structured array with a mixed dtype.

    NOTE : Can be called as `pd_(a, ... )` to emulate pandas dataframes
        You should limit large arrays to a slice ie. a[:50]

    Requires:
    -------
    `a` : array
        A structured/recarray
    `deci` : int
        To facilitate printing, this value is the number of decimal
        points to use for all floating point fields.
    `use_names` : boolean
        If no names are available, then create them
    `prn` : boolean
        True to print, False to return the string
    Notes:
    -----
        `_col_format` : does the actual work of obtaining a representation of
        the column format.

        It is not really possible to deconstruct the exact number of decimals
        to use for float values, so a decision had to be made to simplify.
    """
    dt_names = a.dtype.names
    N = len(dt_names)
    c_names = [["C{:02.0f}".format(i) for i in range(N)], dt_names][use_names]
    # ---- get the column formats from ... _col_format ----
    dts = []
    wdths = []
    pair = list(zip(dt_names, c_names))
    for i in range(len(pair)):
        fld, nme = pair[i]
        c_fmt, col_wdth = _col_format(a[fld], c_name=nme, deci=deci)
        dts.append(c_fmt)
        wdths.append(col_wdth)
    row_frmt = " ".join([('{' + i + '}') for i in dts])
    hdr = ["!s:>" + "{}".format(wdths[i]) for i in range(N)]
    hdr2 = " ".join(["{" + hdr[i] + "}" for i in range(N)])
    header = "--n--" + hdr2.format(*c_names)
    header = "\n{}\n{}".format(header, "-"*len(header))
    txt = [header]
    # ---- check for structured arrays reshaped to (N, 1) instead of (N,) ----
    len_shp = len(a.shape)
    idx = 0
    for i in range(a.shape[0]):
        if len_shp == 1:  # ---- conventional (N,) shaped array
            row = " {:03.0f} ".format(idx) + row_frmt.format(*a[i])
        else:             # ---- reformatted to (N, 1)
            row = " {:03.0f} ".format(idx) + row_frmt.format(*a[i][0])
        idx += 1
        txt.append(row)
    msg = "\n".join([i for i in txt])
    if prn:
        print(msg)
    else:
        return msg


# ----------------------------------------------------------------------
# (5) form_ ... code section .....
#  form_ requires make_row_format
def make_row_format(dim=2, cols=3, a_kind='f', deci=1,
                    a_max=10, a_min=-10, prnt=False):
    """Format the row based on input parameters

    `dim` - int
        Number of dimensions
    `cols` : int
        Columns per dimension

    `a_kind`, `deci`, `a_max` and `a_min` allow you to specify a data type,
    number of decimals and maximum and minimum values to test formatting.
    """
    if a_kind not in ['f', 'i']:
        a_kind = 'f'
    w_, m_ = [[':{}.0f', '{:0.0f}'], [':{}.{}f', '{:0.{}f}']][a_kind == 'f']
    m_fmt = max(len(m_.format(a_max, deci)), len(m_.format(a_min, deci))) + 1
    w_fmt = w_.format(m_fmt, deci)
    row_frmt = ((('{' + w_fmt + '}')*cols + '  ')*dim).strip()
    if prnt:
        frmt = "Row format: dim cols: ({}, {})  kind: {} decimals: {}\n\n{}"
        print(dedent(frmt).format(dim, cols, a_kind, deci, row_frmt))
        a = np.random.randint(a_min, a_max+1, dim*cols)
        col_hdr()  # run col_hdr to produce the column headers
        print(row_frmt.format(*a))
    else:
        return row_frmt


def form_(a, deci=2, wdth=100, title="Array", prefix=". . ", prn=True):
    """Alternate format to frmt_ function.
    Inputs are largely the same.
    """
    def _piece(sub, i, frmt, linewidth):
        """piece together 3D chunks by row"""
        s0 = sub.shape[0]
        block = np.hstack([sub[j] for j in range(s0)])
        txt = ""
        if i is not None:
            fr = (":arr[{}" + ", :{}"*len(a.shape[1:]) + "]\n")
            txt = fr.format(i, *sub.shape)
        for line in block:
            ln = frmt.format(*line)[:linewidth]
            end = ["\n", "...\n"][len(ln) >= linewidth]
            txt += indent(ln + end, ". . ")
        return txt
    # ---- main section ----
    out = "\n{}... ndim: {}  shape: {}\n".format(title, a.ndim, a.shape)
    linewidth = wdth
    if a.ndim <= 1:
        return a
    elif a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    # ---- pull the 1st and 3rd dimension for 3D and 4D arrays
    frmt = make_row_format(dim=a.shape[-3],
                           cols=a.shape[-1],
                           a_kind=a.dtype.kind,
                           deci=deci,
                           a_max=a.max(),
                           a_min=a.min(), prnt=False)
    if a.ndim == 3:
        s0, s1, s2 = a.shape
        out += _piece(a, None, frmt, linewidth)  # ---- _piece ----
    elif a.ndim == 4:
        s0, s1, s2, _ = a.shape
        for i in range(s0):
            out = out + "\n" + _piece(a[i], i, frmt, linewidth)  # ---- _piece
    if prn:
        print(out)
    else:
        return out


# ----------------------------------------------------------------------
# (6)  ---- demos ----
#
def _demo_array():
    """ make an array for testing"""
    sh = [2, 3, 4, 5]
    fac = 1  # 1 for integer, 1.0 for float or 3 for 3x
    a = np.arange(np.prod(sh)).reshape(*sh)*fac
    return a


def _demo_deline():
    """deline array"""
    a = _demo_array()
    deline(a, header="\ndeline demo...", prefix="  .", prn=True)


def _demo_form():
    """demo form_"""
    a = _demo_array()
    form_(a, deci=2, wdth=100, title="form_ demo...", prefix=". . ", prn=True)


def _demo_frmt():
    """frmt_ demo"""
    a = _demo_array()
    frmt_(a, deci=4, wdth=100, title="\nfrmt_demo ...", prn=True)


def _demo_ma():
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
    print("\nArray reshaped to (2, 4, 5)")
    b = a.reshape(2, 4, 5)
    frmt_ma(b, prn=True)


def _demo_rec():
    """load and print a structured array
    """
    pth = _demo_rec.__code__.co_filename  # script path
#    pth = pth.replace("frmts.py", "Data/sample_1000.npy")
    pth = pth.replace("frmts.py", "Data/sample_20.npy")
#    pth = pth.replace("frmts.py", "Data/sample_data.npy")
    aa = np.load(pth)
    fld_names = list(aa.dtype.names)
    cols = min(len(fld_names), 8)
    a = aa[fld_names[:cols]]
    msg = """
    Format ... {}
    record/structured array, with and without field names. """
    print(dedent(msg).format(pth))
    frmt_rec(a[:3], deci=3, use_names=True, prn=True)
    frmt_rec(a[:3], deci=2, use_names=False, prn=True)
    # return a


def _sample_data():
    """just return a recarray
    """
    pth = _sample_data.__code__.co_filename
    # sample_20.npy sample_1000.npy sample_10k.npy sample_100k.npy
    pth = pth.replace("frmts.py", "Data/sample_20.npy")
    a = np.load(pth)
    return a


# -------------------------
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
#    row_frmt = make_row_format()
#    _demo_deline()
#    _demo_frmt()
#    _demo_ma()
#    _demo_rec()
#    _demo_form()
#    a = _sample_data()
