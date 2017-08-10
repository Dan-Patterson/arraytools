# -*- coding: UTF-8 -*-
"""
:Script:   tools.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-08-05
:Purpose:  tools for working with numpy arrays
:Useage:
:  import arraytools as art
:  tools.py and other scripts are part of the array tools package.
:  Access in other programs using .... art.func(params) ....
:Requires:
: import frmts.py which contains and initializes all the format options
:
:Notes:
:-----
:Basic array information:
: - np.typecodes
:   All: '?bhilqpBHILQPefdgFDGSUVOMm'
:   |__AllFloat: 'efdgFDG'
:      |__Float: 'efdg'
:      |__Complex: 'FDG')
:   |__AllInteger: 'bBhHiIlLqQpP'
:   |  |__UnsignedInteger: 'BHILQP'
:   |  |__Integer: 'bhilqp'
:   |__Datetime': 'Mm'
:   |__Character': 'c'
:
: - np.sctypes.keys and np.sctypes.values
:   numpy classes
:   |__complex  complex64, complex128, complex256
:   |__float    float16, float32, float64, float128
:   |__int      int8, int16, int32, int64
:   |__uint     uint8, uint16, uint32, uint63
:   |__others   bool, object, str, void
:
:Functions:  tools function examples below
:---------
: (1) ---- arr2xyz(a, verbose=False) ----
:   - convert an array to x,y,z values, using row/column values for x and y
:   a= np.arange(2*3).reshape(2,3)
:   arr2xyz(a)
:   array([[0, 0, 0],
:          [1, 0, 1],
:          [2, 0, 2],
:          [0, 1, 3],
:          [1, 1, 4],
:          [2, 1, 5]])
:
: (2) ---- block_arr(a, win=[3, 3], nodata=-1) ----
:   - block an array given an input array, a window and a nodata value
:   a = np.arange(16).reshape(4,4)
:   block_arr(a, win=[4, 3], nodata=-1)
:   array([[ 0,  1,  2,  3, -1, -1],
:          [ 4,  5,  6,  7, -1, -1],
:          [ 8,  9, 10, 11, -1, -1],
:          [12, 13, 14, 15, -1, -1]]),
:   masked_array(data =
:       [[[0 1 2]
:         [4 5 6]
:         [8 9 10]
:         [12 13 14]]
:
:        [[3 -- --]
:         [7 -- --]
:         [11 -- --]
:         [15 -- --]]],
:   mask .... snipped ....
:
: (3) ---- change(a, order=[], prn=False) ----
:   - merely a convenience function
:   - a = np.arange(4*5).reshape((4, 5))
:   - change(a, [2, 1, 0, 3, 4])
:   - array([[ 2,  1,  0,  3,  4],
:            [ 7,  6,  5,  8,  9],
:            [12, 11, 10, 13, 14],
:            [17, 16, 15, 18, 19]])
:   - shortcuts
:     b = a[:, [2,1,0,3,4]]   # reorder the columns, keeping the rows
:     c = a[:, [0,2,3]]       # delete columns 1 and 4
:     d = a[[0,1,3,4], :]     # delete row 2, keeping the columns
:     e = a[[0,1,3], [1,2,3]] # keep [0,1],[1,2],[3,3] => ([ 1, 7, 18])
:
: (4) ---- doc_func(func=None) ----
:   see get_func and get_modu
:
: (5) ---- find(a, func, this=None, count=0, keep=[], prn=False, r_lim=2)
:   func - (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
:          (        ==,  !=,  <,   <=,  >,   >=,  >a<, =>a<=,  <a> )
: (5a) --- _func(fn, a, this)
:    called by 'find' see details there
:   (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
: Note: see find1d_demo.py for examples
:
: (6) ---- get_func ----
:    print(art.get_func(art.main))
:    :-----------------------------------------------------------------
:    :Function: .... main ....
:    :Line number... 1334
:    :Docs:
:    Do nothing
:    :Defaults: None
:    :Keyword Defaults: None
:    :Variable names:
:    :Source code:
:       0  def main():
:       1   '''Do nothing'''
:       2      pass
:
: (7) ---- get_modu ----
:
: (8) ---- group_pnts(a, key_fld='ID', keep_flds=['X', 'Y', 'Z'])
: (9) ---- info(a, prn=True) ----
: - example
:   - array([(0, 1, 2, 3, 4), (5, 6, 7, 8, 9),
:            (10, 11, 12, 13, 14), (15, 16, 17, 18, 19)],
:     dtype=[('A', '<i8'), ('B', '<i8')... snip ..., ('E', '<i8')])
:   :---------------------
:   :Array information....
:   :array
:   :  |__shape (4,)
:   :  |__ndim  1
:   :  |__size  4
:   :  |__type  <class 'numpy.ndarray'>
:   :dtype      [('A', '<i8'), ('B', '<i8') ... , ('E', '<i8')]
:   :  |__kind  V
:   :  |__char  V
:   :  |__num   20
:   :  |__type  <class 'numpy.void'>
:   :  |__name  void320
:   :  |__shape ()
:   :  |__description
:   :     |__name, itemsize
:   :     |__['A', '<i8']
:   :     |__['B', '<i8']
:   :     |__['C', '<i8']
:   :     |__['D', '<i8']
:   :     |__['E', '<i8']
:   :---------------------
:
: (10) ---- make_blocks(rows=2, cols=4, r=2, c=2, dt='int')
:    array([[0, 0, 1, 1, 2, 2, 3, 3],
:           [0, 0, 1, 1, 2, 2, 3, 3],
:           [4, 4, 5, 5, 6, 6, 7, 7],
:           [4, 4, 5, 5, 6, 6, 7, 7]])
: (11) ---- make_flds(n=1, names=None, default="col") ----
:   - Example
:   - make_flds(3, names='A,B,C', default='A')
:     =>  dtype([('A', '<f8'), ('B', '<f8'), ('C', '<f8')])
:   - names='A,B,C'
:     easy(f,names)
:     =>  dtype([('A', '<f8'), ('B', '<f8'), ('C', '<f8')])
:   - names='A,B'   # missing a name, so default kicks in
:     easy(f,names,name)
:     =>  dtype([('A', '<f8'), ('B', '<f8'), ('A00', '<f8')]
:
: (12) ---- nd_struct(a) ----
:   - ndarray to structured array
:   (a) keep the dtype the same
:      aa = nd_struct(a)       # produce a structured array from inputs
:      aa.reshape(-1,1)   # structured array
:      array([[(0, 1, 2, 3, 4)],
:             [(5, 6, 7, 8, 9)],
:             [(10, 11, 12, 13, 14)],
:             [(15, 16, 17, 18, 19)]],
:         dtype=[('A', '<i4'), ... snip ... , ('E', '<i4')])
:
:   (b) upcast the dtype
:      a_f = nd_struct(a.astype('float'))  # note astype allows a view
:      array([(0.0, 1.0, 2.0, 3.0, 4.0), ... snip... ,
:             (15.0, 16.0, 17.0, 18.0, 19.0)],
:         dtype=[('A', '<f8'), ... snip ... , ('E', '<f8')])
:
: (13) ---- reclass(z, bins, new_bins, mask=False, mask_val=None)
:   - reclass an array using existing class breaks (bins) and new bins
:     both must be in ascending order
:     z = np.arange(3*5).reshape(3,5)
:     bins = [0, 5, 10, 15]
:     new_bins = [1, 2, 3, 4]
:     z_recl = reclass(z, bins, new_bins, mask=False, mask_val=None)
:     ==> .... z                     ==> .... z_recl
:     array([[ 0,  1,  2,  3,  4],   array([[1, 1, 1, 1, 1],
:            [ 5,  6,  7,  8,  9],          [2, 2, 2, 2, 2],
:            [10, 11, 12, 13, 14]])         [3, 3, 3, 3, 3]])
:
:
: (14) ---- scale(a, x=2, y=2)
:   - scale an array by x, y factors
:     a = np.array([[0, 1, 2], [3, 4, 5]]
:     b = scale(a, x=2, y=2)
:       =  array([[0, 0, 1, 1, 2, 2],
:                 [0, 0, 1, 1, 2, 2],
:                 [3, 3, 4, 4, 5, 5],
:                 [3, 3, 4, 4, 5, 5]])
:   - using scale with np.tile
:     art.scale(a, 2,2)         np.tile(art.scale(a, 2, 2), (2, 2))
:     array([[0, 0, 1, 1],      array([[0, 0, 1, 1, 0, 0, 1, 1],
:            [0, 0, 1, 1],             [0, 0, 1, 1, 0, 0, 1, 1],
:            [2, 2, 3, 3],             [2, 2, 3, 3, 2, 2, 3, 3],
:            [2, 2, 3, 3]])            [2, 2, 3, 3, 2, 2, 3, 3],
:                                      [0, 0, 1, 1, 0, 0, 1, 1],
:                                      [0, 0, 1, 1, 0, 0, 1, 1],
:                                      [2, 2, 3, 3, 2, 2, 3, 3],
:                                      [2, 2, 3, 3, 2, 2, 3, 3]])
:
:
: (15) ---- split_array(a, fld='Id')
:    array 'b'
:    array([(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)],
:          dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')])
:   - split_array(b, fld='A')
:   [array([(0, 1, 2, 3)],
:         dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')]),
:    array([(4, 5, 6, 7)],
:         dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')]),
:    array([(8, 9, 10, 11)],
:         dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')])]
:
: (15) ---- stride(a, r_c=(3, 3))
:   - produce a strided array using a window of r_c shape
:   - calls _check(a, r_c, subok=False) to check for array compliance
:     a =np.arange(15).reshape(3,5)
:     s = stride(a)    stride     ====>   slide    =====>
:     array([[[ 0,  1,  2],  [[ 1,  2,  3],  [[ 2,  3,  4],
:             [ 5,  6,  7],   [ 6,  7,  8],   [ 7,  8,  9],
:             [10, 11, 12]],  [11, 12, 13]],  [12, 13, 14]]])
:
: (16) rolling_stats()... stats for a strided array ...
:    min, max, mean, sum, std, var, ptp
:   (0, 53, 26.5, 1431, 15.5857..., 242.9166..., 53)
:
:References
: - https://github.com/numpy/numpy
: - https://github.com/numpy/numpy/blob/master/numpy/lib/_iotools.py
: - https://github.com/numpy/numpy/blob/master/numpy/lib/stride_tricks.py
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
from numpy.lib.stride_tricks import as_strided
from textwrap import dedent, indent

__all__ = ['arr2xyz',
           'block_arr',
           'change',
           'doc_func',
           'find',
           'get_func',
           'get_modu',
           'group_pnts',
           'info',
           'make_blocks',
           'make_flds',
           'nd_struct',
           'reclass',
           'scale',
           'stride',
           'rolling_stats'
           ]
__xtras__ = ['_func', '_check',
             '_demo', 'run_deco',
             'time_deco'
             ]

__outside__ = ['dedent', 'indent']

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ---- decorators ----

def time_deco(func):  # timing originally
    """timing decorator function
    :print("\n  print results inside wrapper or use <return> ... ")
    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()        # start time
        result = func(*args, **kwargs)  # ... run the function ...
        t1 = time.perf_counter()        # end time
        dt = t1-t0
        print("\nTiming function for... {}\n".format(func.__name__))
        print("  Time: {: <8.2e}s for {:,} objects".format(dt, len(result)))
        return result                   # return the result of the function
        return dt                       # return delta time
    return wrapper


def run_deco(func):
    """Prints basic function information and the results of a run.
    :Requires:  from functools import wraps
    :  Uncomment the import or move it to within the script.
    :Useage:   @func_run  on the line above the function
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        frmt = "\n".join(["Function... {}", "  args.... {}",
                          "  kwargs.. {}", "  docs.... {}"])
        ar = [func.__name__, args, kwargs, func.__doc__]
        print(dedent(frmt).format(*ar))
        result = func(*args, **kwargs)
        print("{!r:}\n".format(result))  # comment out if results not needed
        return result                    # for optional use outside.
    return wrapper


# ---- functions ----

# ----------------------------------------------------------------------
# (1) arr2xyz .... code section
def arr2xyz(a, verbose=False):
    """Produce an array such that the row, column values are used for x,y
    :  and array values for z.
    :Returns:
    :--------
    :  a mesh grid with values, dimensions and shapes are changed so
    :  that ndim=2, ie shape(3,4,5), ndim=3 becomes shape(12,5), ndim=2
    """
    if a.ndim == 1:
        a = a.reshape(a.shape[0], 1)
    if a.ndim > 2:
        a = a.reshape(np.product(a.shape[:-1]), a.shape[-1])
    r, c = a.shape
    XX, YY = np.meshgrid(np.arange(c), np.arange(r))
    tbl = np.vstack((XX.ravel(), YY.ravel(), a.ravel())).T
    return tbl


# ----------------------------------------------------------------------
# (2) block_arr .... code section
def block_arr(a, win=[3, 3], nodata=-1):
    """Block array into window sized chunks padding to the right and bottom
    :  to accommodate array and window shape.
    :Requires:
    :--------
    :  a - 2D array
    :  win = [rows, cols], aka y,x, m,n sized window
    :  nodata - to use for the mask
    :Returns:
    :-------
    :  the padded array and the masked array blocked
    :Reference:
    :---------
    :  http://stackoverflow.com/questions/40275876/
    :         how-to-reshape-this-image-array-in-python
    :  def block_2(a, blocks=2)
    :      B = blocks # Blocksize
    :      m, n = a.shape
    :      out = a.reshape(m//B, B, n//B, B).swapaxes(1, 2).reshape(-1, B, B)
    :      return out
    :
    """
    s = np.array(a.shape)
    if len(win) != 2:
        print("\n....... Read the docs .....\n{}".format(block_arr.__doc__))
        return None
    win = np.asarray(win)
    m = divmod(s, win)
    s2 = win*m[0] + win*(m[1] != 0)
    ypad, xpad = s2 - a.shape
    pad = ((0, ypad), (0, xpad))
    p_with = ((nodata, nodata), (nodata, nodata))
    b = np.pad(a, pad_width=pad, mode='constant', constant_values=p_with)
    w_y, w_x = win       # Blocksize
    y, x = b.shape       # padded array
    c = b.reshape((y//w_y, w_y, x//w_x, w_x))
    c = c.swapaxes(1, 2).reshape(-1, w_y, w_x)
    c = np.ma.masked_equal(c, nodata)
    c.set_fill_value(nodata)
    return c


# ----------------------------------------------------------------------
# (3) change ... code section .....
def change(a, order=[], prn=False):
    """Reorder and/or drop columns in an array using slicing.
    :  Fields not included will be dropped in the output array.
    :
    :Requires:
    :--------
    :  For a structured/recarray, the desired field order is required.
    :  An ndarray, not using named fields, will require the numerical
    :  order of the fields.
    :
    : ---- remove fields ----
    :  To remove fields, simply leave them out of the list.  The
    :  order of the remaining fields will be reflected in the output.
    :
    :  This is a convenience function.... see the module header for
    :  one-liner syntax.
    :
    :  use:  arr_info(a, verbose=True)
    :        This gives field names which can be copied for use here.
    """
    names = a.dtype.names
    if names is None:
        b = a[:, order]
    else:
        out_flds = []
        out_flds = [i for i in order if i in names]
        if prn:
            missing = [i for i in names if i not in order]
            missing.extend([i for i in order if i not in out_flds])
            frmt = """
            : change(a)
            : - field(s) {}
            : - not found, missing or removed.
            """
            print(dedent(frmt).format(missing))
        b = a[out_flds]
    return b


# ----------------------------------------------------------------------
# (4) doc_func ... code section ...
def doc_func(func=None):
    """(doc_func)...Documenting code using inspect
    :Requires:
    :--------
    :  import inspect  # module
    :Source code for...
    :  module level   => inspect.getsourcelines(sys.modules[__name__])[0]
    :  function level
    :       as a list => inspect.getsourcelines(num_41)[0]
    :     as a string => inspect.getsource(num_41)
    :  file level => script = sys.argv[0]
    :Returns:  a listing of the source code with line numbers
    :-------
    :
    """
    def demo_func():
        """dummy...
        : Demonstrates retrieving and documenting module and function info.
        """
        def sub():
            """sub in dummy"""
            pass
        return None
    import inspect
    if func is None:
        func = demo_func
    script = sys.argv[0]  # a useful way to get a file's name
    lines, line_num = inspect.getsourcelines(func)
    code = "".join(["{:4d}  {}".format(idx, line)
                    for idx, line in enumerate(lines)])
    args = [line_num, code,
            inspect.getcomments(func), inspect.isfunction(func),
            inspect.ismethod(func), inspect.getmodulename(script)
            ]
    frmt = """
    :----------------------------------------------------------------------
    : ---- doc_func(func) ----
    :Code for a function on line...{}...
    {}
    :Comments preceeding function
    {}
    :function?... {} ... or method? {}
    :Module name...
    {}
    :
    :----------------------------------------------------------------------
    """
    out = dedent(frmt).format(*args)
    return out


# ----------------------------------------------------------------------
# (5, 5a) find .... code section
def _func(fn, a, this):
    """called by 'find' see details there
    :  (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
    """
    #
    fn = fn.lower().strip()
    if fn in ['cumsum', 'csum', 'cu']:
        return np.where(np.cumsum(a) <= this)[0]
    if fn in ['eq', 'e', '==']:
        return np.where(np.in1d(a, this))[0]
    if fn in ['neq', 'ne', '!=']:
        return np.where(~np.in1d(a, this))[0]  # (a, this, invert=True)
    if fn in ['ls', 'les', '<']:
        return np.where(a < this)[0]
    if fn in ['lseq', 'lese', '<=']:
        return np.where(a <= this)[0]
    if fn in ['gt', 'grt', '>']:
        return np.where(a > this)[0]
    if fn in ['gteq', 'gte', '>=']:
        return np.where(a >= this)[0]
    if fn in ['btwn', 'btw', '>a<']:
        low, upp = this
        return np.where((a >= low) & (a < upp))[0]
    if fn in ['btwni', 'btwi', '=>a<=']:
        low, upp = this
        return np.where((a >= low) & (a <= upp))[0]
    if fn in ['byond', 'bey', '<a>']:
        low, upp = this
        return np.where((a < low) | (a > upp))[0]


# @time_deco
def find(a, func, this=None, count=0, keep=[], prn=False, r_lim=2):
    """
    : a    - array or array like
    : func - (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
    :        (        ==,  !=,  <,   <=,  >,   >=,  >a<, =>a<=,  <a> )
    : count - only used for recursive functions
    : keep - for future use
    : verbose - True for test printing
    : max_depth - prevent recursive functions running wild, it can be varied
    :
    : recursive functions:
    : cumsum
    :   An example of using recursion to split a list/array of data
    :   parsing the results into groups that sum to this.  For example,
    :   split input into groups where the total population is less than
    :   a threshold (this).  The default is to use a sequential list,
    :   however, the inputs could be randomized prior to running.
    :Returns:
    :-------
    : a 1D or 2D array meeting the conditions
    :
    """
    a = np.asarray(a)              # ---- ensure array format
    this = np.asarray(this)
    if prn:                        # ---- optional print
        print("({}) Input values....\n  {}".format(count, a))
    ix = _func(func, a, this)      # ---- sub function -----
    if ix is not None:
        keep.append(a[ix])         # ---- slice and save
        if len(ix) > 1:
            a = a[(len(ix)):]      # ---- use remainder
        else:
            a = a[(len(ix)+1):]
    if prn:                        # optional print
        print("  Remaining\n  {}".format(a))
    # ---- recursion functions check and calls ----
    if func in ['cumsum']:  # functions that support recursion
        if (len(a) > 0) and (count < r_lim):  # recursive call
            count += 1
            find(a, func, this, count, keep, prn, r_lim)
        elif (count == r_lim):
            frmt = """Recursion check... count {} == {} recursion limit
                   Warning...increase recursion limit, reduce sample size\n
                   or changes conditions"""
            print(dedent(frmt).format(count, r_lim))
    # ---- end recursive functions ----
    # print("keep for {} : {}".format(func,keep))
    #
    if len(keep) == 1:   # for most functions, this is it
        final = keep[0]
    else:                # for recursive functions, there will be more
        temp = []
        incr = 0
        for i in keep:
            temp.append(np.vstack((i, np.array([incr]*len(i)))))
            incr += 1
        temp = (np.hstack(temp)).T
        dt = [('orig', '<i8'), ('class', '<i8')]
        final = np.zeros((temp.shape[0],), dtype=dt)
        final['orig'] = temp[:, 0]
        final['class'] = temp[:, 1]
        # ---- end recursive section
    return final


# ----------------------------------------------------------------------
# (6) get_func .... code section
def get_func(obj, line_nums=True, verbose=True):
    """Get function (def) information.
    :Requires:
    :--------
    :  from textwrap import dedent, indent, wrap
    :  import inspect
    :Returns:
    :-------
    :  The function information includes arguments and source code.
    :  A string is returned for printing.
    """
    frmt = """
    :-----------------------------------------------------------------
    :Function: .... {} ....
    :Line number... {}
    :Docs:
    {}
    :Defaults: {}
    :Keyword Defaults: {}
    :Variable names:
    {}
    :Source code:
    {}
    :
    :-----------------------------------------------------------------
    """
    import inspect
    from textwrap import dedent, wrap
    lines, ln_num = inspect.getsourcelines(obj)
    if line_nums:
        code = "".join(["{:4d}  {}".format(idx, line)
                        for idx, line in enumerate(lines)])
    else:
        code = "".join(["{}".format(line) for line in lines])
    vars = ", ".join([i for i in obj.__code__.co_varnames])
    vars = wrap(vars, 50)
    vars = "\n".join([i for i in vars])
    args = [obj.__name__, ln_num, dedent(obj.__doc__), obj.__defaults__,
            obj.__kwdefaults__, indent(vars, "    "), code]
    code_mem = dedent(frmt).format(*args)
    return code_mem


# ----------------------------------------------------------------------
# (7) get_modu .... code section
def get_modu(obj):
    """Get module (script) information, including source code for
    :  documentation purposes.
    :Requires:
    :--------
    :  from textwrap import dedent, indent
    :  import inspect
    :Returns:
    :-------
    :  A string is returned for printing.
    """
    frmt = """
    :-----------------------------------------------------------------
    :Module: .... {} ....
    :------
    :File: ......
    {}\n
    :Docs:
    {}\n
    :Members:
    {}\n
    :Source code:
    {}
    :
    :-----------------------------------------------------------------
    """
    import inspect
    from textwrap import dedent
    lines, line_num = inspect.getsourcelines(obj)
    memb = [i[0] for i in inspect.getmembers(obj)]
    code = "".join(["{:4d}  {}".format(idx, line)
                    for idx, line in enumerate(lines)])
    args = [obj.__name__, obj.__file__, obj.__doc__, memb, code]
    mod_mem = dedent(frmt).format(*args)
    return mod_mem


# ----------------------------------------------------------------------
# (8)
def group_pnts(a, key_fld='ID', keep_flds=['X', 'Y', 'Z']):
    """Group points for a feature that has been exploded to points by
    :  arcpy.da.FeatureClassToNumPyArray.
    :Requires:
    :---------
    : a - a structured array, assuming ID, X, Y, {Z} and whatever else
    :   - the array is assumed to be sorted... which will be the case
    :Returns:
    :--------
    : see np.unique descriptions below
    :References:
    :-----------
    :  https://jakevdp.github.io/blog/2017/03/22/group-by-from-scratch/
    :  http://esantorella.com/2016/06/16/groupby/
    :Notes:
    :------ split-apply-combine
    """
    returned = np.unique(a[key_fld],           # the unique id field
                         return_index=True,    # first occurrence index
                         return_inverse=True,  # indices needed to remake array
                         return_counts=True)   # number in each group
    uniq, idx, inv, cnt = returned
    from_to = [[idx[i-1], idx[i]] for i in range(1, len(idx))]
    subs = [a[keep_flds][i:j] for i, j in from_to]
    groups = [sub.view(dtype='float').reshape(sub.shape[0], -1)
               for sub in subs]
    return groups


# ----------------------------------------------------------------------
# (8) info .... code section
def info(a, prn=True):
    """Returns basic information about an numpy array.
    :Requires:
    :--------
    : a - an array
    :Return example:
    :--------------
    :  a = np.arange(2.*3.).reshape(2,3) # quick float64 array
    :  arr_info(a)
    : ---------------------
    : Array information....
    : array
    :   |__shape (2, 3)
    :   |__ndim  2
    :   |__size  6
    :   |__bytes
    :   |__type  <class 'numpy.ndarray'>
    :   |__strides  (24, 8)
    : dtype      float64
    :   |__kind  f
    :   |__char  d
    :   |__num   12
    :   |__type  <class 'numpy.float64'>
    :   |__name  float64
    :   |__shape ()
    :   |__description
    :      |__name, itemsize
    :      |__['', '<f8']
    : ---------------------
    :
    """
    if not isinstance(a, (np.ndarray, np.ma.core.MaskedArray)):
        print("Requires a numpy ndarray")
        return "Read the docs"
    frmt = """
    :---------------------
    :Array information....
    :array
    :  |__shape {}\n    :  |__ndim  {}\n    :  |__size  {}
    :  |__bytes {}\n    :  |__type  {}\n    :  |__strides  {}
    :dtype      {}
    :  |__kind  {}\n    :  |__char  {}\n    :  |__num   {}
    :  |__type  {}\n    :  |__name  {}\n    :  |__shape {}
    :  |__description
    :     |__name, itemsize"""
    dt = a.dtype
    info = [a.shape, a.ndim, a.size, a.nbytes, type(a), a.strides, dt,
            dt.kind, dt.char, dt.num, dt.type, dt.name, dt.shape]
    flds = sorted([[k, v] for k, v in dt.descr])
    out = dedent(frmt).format(*info) + "\n"
    leader = "".join([":     |__{}\n".format(i) for i in flds])
    leader = leader + ":---------------------"
    out = out + leader
    if prn:
        print(out)
    else:
        return out


# ----------------------------------------------------------------------
# (9) make_blocks ... code section .....
def make_blocks(rows=3, cols=3, r=2, c=2, dt='int'):
    """Make a block array with rows * cols containing r*c sub windows
    :Requires:
    :---------
    :  specify the rows, columns, then the block size, r,c and dtype
    :Returns:
    :--------
    :  The defaults produce an 8 column by 8 row array numbered from
    :  0 to (rows*cols) - 1
    :  array.shape = (rows * r, cols * c) = (6, 6)
    :Notes:
    :-----
    :  scale - if you want specific values during array construction.
    """
    a = np.arange(rows*cols, dtype=dt).reshape(rows, cols)
    a = scale(a, x=r, y=c)
    return a


# ----------------------------------------------------------------------
# (10) make_flds .... code section
def make_flds(n=1, as_type='float', names=None, def_name="col"):
    """Create float fields for statistics and their names.
    :Requires:
    :--------
    : n    - number of fields to create excluding the names field
    : def_name - base name to use, numeric values will be produced for each
    :        dimension for the 3D array, ie Values_00... Values_nn
    : from numpy.lib._iotools import easy_dtype as easy
    :    Don't forget the above, a cool way to create fields quickly
    :Returns:
    :-------
    : - a dtype, which contains the necessary fields
    :   to contain the float values
    """
    from numpy.lib._iotools import easy_dtype as easy
    if as_type in ['float', 'f8', '<f8']:
        as_type = '<f8'
    elif as_type in ['int', 'i4', 'i8', '<i4', '<i8']:
        as_type = '<i8'
    else:
        as_type = 'str'
    f = ",".join([as_type for i in range(n)])
    names = ", ".join(["{}_{:>02}".format(def_name, i) for i in range(n)])
    if names is None:
        dt = easy(f, names=names)
    else:
        dt = easy(f, names=names, defaultfmt=def_name)
    return dt


# ----------------------------------------------------------------------
# (11) nd_struct .... code section
def nd_struct(a):
    """ convert ndarray to structured/recarray
    :Requires:
    :--------
    : a - ndarray with a uniform dtype, the field names are assigned
    :     from an alphabetical list up to 52 fields
    :   - the dtype of the input array is retained, but can be upcast
    :Returns:
    : ---- examples ----
    :  - a = np.arange(2*3).reshape(2,3)
    :  - array([[0, 1, 2],
    :           [3, 4, 5]])  # dtype('int64')
    :  - b = a.astype('float64')
    :    array([[ 0.000,  1.000,  2.000],
    :           [ 3.000,  4.000,  5.000]])
    :  - c = nd_struct(a)
    :    array([(0, 1, 2), (3, 4, 5)],
    :         dtype=[('A', '<i8'), ('B', '<i8'), ('C', '<i8')])
    :  - d = nd_struct(b.astype('float64'))
    :    array([(0.0, 1.0, 2.0), (3.0, 4.0, 5.0)],
    :         dtype=[('A', '<f8'), ('B', '<f8'), ('C', '<f8')])
    :-------
    """
    alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    if a.ndim != 2:
        frmt = "Wrong array shape... read the docs..\n{}"
        print(frmt.format(nd_struct.__doc__))
        return a
    rows, cols = a.shape
    dt_name = a.dtype.descr[0][1]  # a.dtype.name
    fld_names = list(alph)[:cols]
    dt = [(i, dt_name) for i in fld_names]
    aa = np.zeros((rows,), dtype=dt)
    names = aa.dtype.names
    for i in range(a.shape[1]):
        aa[names[i]] = a[:, i]
    return aa


# ----------------------------------------------------------------------
# (12) reclass .... code section
def reclass(a, bins=[], new_bins=[], mask=False, mask_val=None):
    """Reclass an array of integer or floating point values.
    :Requires:
    :--------
    : bins - sequential list/array of the lower limits of each class
    :        include one value higher to cover the upper range.
    : new_bins - new class values for each bin
    : mask - whether the raster contains nodata values or values to
    :        be masked with mask_val
    : Array dimensions will be squeezed.
    :Example:
    :-------
    :  z = np.arange(3*5).reshape(3,5)
    :  bins = [0, 5, 10, 15]
    :  new_bins = [1, 2, 3, 4]
    :  z_recl = reclass(z, bins, new_bins, mask=False, mask_val=None)
    :  ==> .... z                     ==> .... z_recl
    :  array([[ 0,  1,  2,  3,  4],   array([[1, 1, 1, 1, 1],
    :         [ 5,  6,  7,  8,  9],          [2, 2, 2, 2, 2],
    :         [10, 11, 12, 13, 14]])         [3, 3, 3, 3, 3]])
    """
    a_rc = np.zeros_like(a)
    if (len(bins) < 2):  # or (len(new_bins <2)):
        print("Bins = {} new = {} won't work".format(bins, new_bins))
        return a
    if len(new_bins) < 2:
        new_bins = np.arange(1, len(bins)+2)
    new_classes = list(zip(bins[:-1], bins[1:], new_bins))
    for rc in new_classes:
        q1 = (a >= rc[0])
        q2 = (a < rc[1])
        a_rc = a_rc + np.where(q1 & q2, rc[2], 0)
    return a_rc


# ----------------------------------------------------------------------
# (13) scale .... code section
def scale(a, x=2, y=2, num_z=None):
    """Scale the input array repeating the array values up by the
    :  x and y factors.
    :Requires:
    :--------
    : a - an ndarray, 1D arrays will be upcast to 2D
    : x, y - factors to scale the array in x (col) and y (row)
    :      - scale factors must be greater than 2
    : num_z - for 3D, produces the 3rd dimension, ie. if num_z = 3 with the
    :    defaults, you will get an array with shape=(3, 6, 6)
    : how - if num_z != None or 0, then the options are
    :    'repeat', 'random'.  With 'repeat' the extras are kept the same
    :     and you can add random values to particular slices of the 3rd
    :     dimension, or multiply them etc etc.
    :Returns:
    :-------
    : a = np.array([[0, 1, 2], [3, 4, 5]]
    : b = scale(a, x=2, y=2)
    :   =  array([[0, 0, 1, 1, 2, 2],
    :             [0, 0, 1, 1, 2, 2],
    :             [3, 3, 4, 4, 5, 5],
    :             [3, 3, 4, 4, 5, 5]])
    :Notes:
    :-----
    :  a=np.arange(2*2).reshape(2,2)
    :  a = array([[0, 1],
    :             [2, 3]])
    :  f_(scale(a, x=2, y=2, num_z=2))
    :  Array... shape (3, 4, 4), ndim 3, not masked
    :   0, 0, 1, 1    0, 0, 1, 1    0, 0, 1, 1
    :   0, 0, 1, 1    0, 0, 1, 1    0, 0, 1, 1
    :   2, 2, 3, 3    2, 2, 3, 3    2, 2, 3, 3
    :   2, 2, 3, 3    2, 2, 3, 3    2, 2, 3, 3
    :   sub (0)       sub (1)       sub (2)
    :--------
    """
    if (x < 1) or (y < 1):
        print("x or y scale < 1... read the docs".format(scale.__doc__))
        return None
    a = np.atleast_2d(a)
    z0 = np.tile(a.repeat(x), y)  # repeat for x, then tile
    z1 = np.hsplit(z0, y)         # split into y parts horizontally
    z2 = np.vstack(z1)            # stack them vertically
    if a.shape[0] > 1:            # if there are more, repeat
        z3 = np.hsplit(z2, a.shape[0])
        z3 = np.vstack(z3)
    else:
        z3 = np.vstack(z2)
    if num_z not in (0, None):
        d = [z3]
        for i in range(num_z):
            d.append(z3)
        z3 = np.dstack(d)
        z3 = np.rollaxis(z3, 2, 0)
    return z3


# ----------------------------------------------------------------------
# (14)  ---- move get_func and get_modu out
def split_array(a, fld='ID'):
    """Split a structured or recarray array using unique values in the
    :  'fld' field.  It is assumed that there is a sequential ordering to
    :  the values in the field.  If there is not, use np.where in conjunction
    :  with np.unique or sort the array first.
    :
    :Requires:
    :--------
    : a   - a structured or recarray
    : fld - a numeric field assumed to be sorted which indicates which group
    :       a record belongs to.
    :
    :Returns:
    :-------
    : - a list of arrays split on the categorizing field
    """
    return np.split(a, np.where(np.diff(a[fld]))[0] + 1)


# ----------------------------------------------------------------------
# (15) stride .... code section
def _check(a, r_c, subok=False):
    """Performs the array checks necessary for stride and block.
    : a   - Array or list.
    : r_c - tuple/list/array of rows x cols.
    : subok - from numpy 1.12 added, keep for now
    :Returns:
    :------
    :Attempts will be made to ...
    :  produce a shape at least (1*c).  For a scalar, the
    :  minimum shape will be (1*r) for 1D array or (1*c) for 2D
    :  array if r<c.  Be aware
    """
    if isinstance(r_c, (int, float)):
        r_c = (1, int(r_c))
    r, c = r_c
    if a.ndim == 1:
        a = np.atleast_2d(a)
    r, c = r_c = (min(r, a.shape[0]), min(c, a.shape[1]))
    a = np.array(a, copy=False, subok=subok)
    return a, r, c, tuple(r_c)


def _pad(a, nan_edge=False):
    """Pad a sliding array to allow for stats"""
    if nan_edge:
        a = np.pad(a, pad_width=(1, 2), mode="constant",
                   constant_values=(np.NaN, np.NaN))
    else:
        a = np.pad(a, pad_width=(1, 1), mode="reflect")
    return a


def stride(a, r_c=(3, 3)):
    """Provide a 2D sliding/moving view of an array.
    :  There is no edge correction for outputs.
    :
    :Requires:
    :--------
    : _check(a, r_c) ... Runs the checks on the inputs.
    : a - array or list, usually a 2D array.  Assumes rows is >=1,
    :     it is corrected as is the number of columns.
    : r_c - tuple/list/array of rows x cols.  Attempts  to
    :     produce a shape at least (1*c).  For a scalar, the
    :     minimum shape will be (1*r) for 1D array or 2D
    :     array if r<c.  Be aware
    """
    a, r, c, r_c = _check(a, r_c)
    shape = (a.shape[0] - r + 1, a.shape[1] - c + 1) + r_c
    strides = a.strides * 2
    a_s = (as_strided(a, shape=shape, strides=strides)).squeeze()
    return a_s


# ----------------------------------------------------------------------
# (16) stride .... code section
def rolling_stats(a, no_null=True, prn=True):
    """Statistics on the last two dimensions of an array.
    :Requires
    :--------
    : a - 2D array
    : no_null - boolean, whether to use masked values (nan) or not.
    : prn - boolean, to print the results or return the values.
    :
    :Returns
    :-------
    : The results return an array of 4 dimensions representing the original
    : array size and block size
    : eg.  original = 6x6 array   block = 3x3
    :      breaking the array into 4 chunks
    """
    a = np.asarray(a)
    a = np.atleast_2d(a)
    ax = None
    if a.ndim > 1:
        ax = tuple(np.arange(len(a.shape))[-2:])
    if no_null:
        a_min = a.min(axis=ax)
        a_max = a.max(axis=ax)
        a_mean = a.mean(axis=ax)
        a_sum = a.sum(axis=ax)
        a_std = a.std(axis=ax)
        a_var = a.var(axis=ax)
        a_ptp = a_max - a_min
    else:
        a_min = np.nanmin(a, axis=(ax))
        a_max = np.nanmax(a, axis=(ax))
        a_mean = np.nanmean(a, axis=(ax))
        a_sum = np.nansum(a, axis=(ax))
        a_std = np.nanstd(a, axis=(ax))
        a_var = np.nanvar(a, axis=(ax))
        a_ptp = a_max - a_min
    if prn:
        frmt = "Minimum...\n{}\nMaximum...\n{}\nMean...\n{}\n" + \
               "Sum...\n{}\nStd...\n{}\nVar...\n{}\nRange...\n{}"
        frmt = dedent(frmt)
        args = [a_min, a_max, a_mean, a_sum, a_std, a_var, a_ptp]
        print(frmt.format(*args))
    else:
        return a_min, a_max, a_mean, a_sum, a_std, a_var, a_ptp


# ----------------------------------------------------------------------
# _help .... code section
def _help():
    """arraytools"""
    _hf = """
    :-------------------------------------------------------------------:
    : ---- arrtools functions  (loaded as 'art') ----
    : ---- from tools.py
    (1)  arr2xyz(a, verbose=False)
         array (col, rows) to (x, y) and array values for z.
    (2)  block_arr(a, win=[3, 3], nodata=-1)
         break an array up into blocks
    (3)  change(a, order=[], prn=False)
         reorder and/or drop columns
    (4)  doc_func(func=None)
         Documenting code using inspect
    (5)  find(a, func, this=None, count=0, keep=[], prn=False, r_lim=2)
         find elements in an array using...
         func - (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
               (      , ==,  !=,  <,   <=,  >,   >=,  >a<, =>a<=,  <a> )
    (6)  get_func(obj, line_nums=True, verbose=True)
         pull in function code
    (7)  get_modu(obj)
         pull in module code
    (8)  group_pnts(a, key_fld='ID', keep_flds=['X', 'Y', 'Z'])
    (9)  info(a)  array info
    (10) make_blocks(rows=3, cols=3, r=2, c=2, dt='int')
         make arrays consisting of blocks
    (11) make_flds(n=1, as_type='float', names=None, def_name='col')
         make structured/recarray fields
    (12) nd_struct(a)
         convert an ndarray to a structured array with fields
    (13) reclass(a, bins=[], new_bins=[], mask=False, mask_val=None)
         reclass an array
    (14) scale(a, x=2, y=2, num_z=None)
         scale an array up in size by repeating values
    (15) split_array(a, fld='ID')
         split an array using an index field
    (16) stride(a, r_c=(3, 3))
         stride an array for moving window functions
    (17) rolling_stats((a0, no_null=True, prn=True))
    :-------------------------------------------------------------------:
    """
    print(dedent(_hf))


# ----------------------------------------------------------------------
# _demo .... code section
# @run_deco
def _demo():
    """
    : - Run examples of the existing functions.
    """
    a = np.arange(3*4).reshape(3, 4)
    b = nd_struct(a)
    c = np.arange(2*3*4).reshape(2, 3, 4)
    d = np.arange(9*6).reshape(9, 6)
    bloc = block_arr(a, win=[2, 2], nodata=-1)  # for block
    chng = change(b, order=['B', 'C', 'A'], prn=False)
    docf = doc_func(stride)
    scal = scale(a, 2)
    a_inf = info(d, prn=False)
    m_blk = make_blocks(rows=3, cols=3, r=2, c=2, dt='int')
    m_fld = str(make_flds(n=3, as_type='int', names=["A", "B", "C"]))
    spl = split_array(b, fld='A')
    stri = stride(a, (3, 3))
    rsta = rolling_stats(d, no_null=True, prn=False)
    frmt = """
: ----- _demo {}
:
:Input ndarray, 'a' ...
{!r:}\n
:Input ndarray, 'b' ...
{!r:}\n
:Input ndarray, 'c' ...
{!r:}\n
:Input ndarray, 'd' ...
{!r:}\n
:---- Functions by number  ---------------------------------------------
:(1)  arr2xyz(a, verbose=False)
{}\n
:(2)  block_arr(a, win=[2, 2], nodata=-1)
{}\n
:(3) change(b, order=['B', 'C', 'A'], prn=False
:    Array 'b', reordered with 2 fields dropped...
{!r:}\n
:(3) doc_func(col_hdr) ... documenting a function...
{}\n
:(4) scale() ... scale an array up by an integer factor...
{}\n
:(5) array info ... info(a)
{}\n
:(6) make_flds() ... create default field names ...
{}\n
:(7) split_array() ... split an array according to an index field
{}\n
:(8) stride() ... stride an array ....
{}\n
:(9) make_blocks(rows=3, cols=3, r=2, c=2, dt='int')
{}\n
:(10) nd_struct() ... make a structured array from another array ...
{!r:}\n
:(11) rolling_stats()... stats for a strided array ...
:    min, max, mean, sum, std, var, ptp
{}
"""
    args = ["-"*62, a, b, c, d,
            arr2xyz(a),
            bloc,
            chng.reshape(a.shape[0], -1),
            docf,
            scal,
            a_inf,
            m_fld,
            spl,
            stri,
            m_blk,
            nd_struct(a),
            rsta]  # f21

    print(frmt.format(*args))
    # del args, d, e


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
    _demo()
