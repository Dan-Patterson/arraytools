# -*- coding: UTF-8 -*-
"""
:Script:   tools.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-01-27
:Purpose:  tools for working with numpy arrays
:Useage:
:  import arraytools as art
:  tools.py and other scripts are part of the array tools package.
:  Access in other programs using .... art.func(params) ....
:
:Requires:
:--------
: see import section and __init__.py
:
:Notes:
:-----
:Basic array information:
: - np.typecodes ie np.typecodes.items()... np.typecodes['AllInteger']
:   All: '?bhilqpBHILQPefdgFDGSUVOMm'
:   |__AllFloat: 'efdgFDG'
:      |__Float: 'efdg'
:      |__Complex: 'FDG'
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
:Numbers:
:  np.inf, -np.inf
:  np.iinfo(np.int8).min  or .max = -128, 128
:  np.iinfo(np.int16).min or .max = -32768, 32768
:  np.iinfo(np.int32) - iinfo(min=-2147483648, max=2147483647, dtype=int32)
:  np.finfo(np.float64)
:  np.finfo(resolution=1e-15, min=-1.7976931348623157e+308,
:           max=1.7976931348623157e+308, dtype=float64)
:
:Functions:  tools function examples below
:---------
: (1) ---- doc_func(func=None) ----
:   see get_func and get_modu
:
: (2) ---- get_func ----
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
: (3) ---- get_modu ----
:
: (4) ---- info(a, prn=True) ----
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
: (5) ---- num_to_nan
: (5) ---- num_to_mask
:
: (6) ---- make_blocks(rows=2, cols=4, r=2, c=2, dt='int')
:    array([[0, 0, 1, 1, 2, 2, 3, 3],
:           [0, 0, 1, 1, 2, 2, 3, 3],
:           [4, 4, 5, 5, 6, 6, 7, 7],
:           [4, 4, 5, 5, 6, 6, 7, 7]])
:
: (7) ---- make_flds(n=1, names=None, default="col") ----
:   - Example
:   - make_flds(3, names='A,B,C', default='A')
:     =>  dtype([('A', '<f8'), ('B', '<f8'), ('C', '<f8')])
:   - names='A,B,C'
:     easy(f,names)
:     =>  dtype([('A', '<f8'), ('B', '<f8'), ('C', '<f8')])
:   - names='A,B'   # missing a name, so default kicks in
:     easy(f,names,name)
:     =>  dtype([('A', '<f8'), ('B', '<f8'), ('A00', '<f8')]
: (8) rec_arr
:
: (9) ---- arr2xyz(a, verbose=False) ----
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
: (10) ---- change_arr(a, order=[], prn=False) ----
:   - merely a convenience function
:   - a = np.arange(4*5).reshape((4, 5))
:   - change(a, [2, 1, 0, 3, 4])
:   - array([[ 2,  1,  0,  3,  4],
:            [ 7,  6,  5,  8,  9],
:            [12, 11, 10, 13, 14],
:            [17, 16, 15, 18, 19]])
:   - shortcuts
:     b = a[:, [2, 1, 0, 3, 4]]    # reorder the columns, keeping the rows
:     c = a[:, [0, 2, 3]]          # delete columns 1 and 4
:     d = a[[0, 1, 3, 4], :]       # delete row 2, keeping the columns
:     e = a[[0, 1, 3], [1, 2, 3]]  # keep [0, 1], [1, 2], [3, 3]
:                                    => ([ 1, 7, 18])
:
: (11) ---- nd2struct(a) ----
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
: (12) ---- scale(a, x=2, y=2)
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
: (13) ---- split_array(a, fld='Id')
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
: (14) ---- _pad_(a, pad_with=None, size=(1, 1))
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
:    - _pad  to pad an array prior to striding or blocking
:    - block  calls stride with non-overlapping blocks with no padding
:
: (16) ---- block(a, win=(3, 3))
:
:
: (17) ---- block_arr(a, win=[3, 3], nodata=-1) ----
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
: (18) ---- find(a, func, this=None, count=0, keep=[], prn=False, r_lim=2)
:   func - (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
:          (        ==,  !=,  <,   <=,  >,   >=,  >a<, =>a<=,  <a> )
: (18a) --- _func(fn, a, this)
:    called by 'find' see details there
:   (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
: Note: see find1d_demo.py for examples
:
: (19) ---- group_pnts(a, key_fld='ID', keep_flds=['X', 'Y', 'Z'])
:
: (20) ---- group_vals(seq, stepsize=1)
:   - seq = [1, 2, 4, 5, 8, 9, 10]
:     stepsize = 1
:   - [array([1, 2]), array([4, 5]), array([ 8,  9, 10])]
:
: (21) ---- reclass(z, bins, new_bins, mask=False, mask_val=None)
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
: (22) rolling_stats()... stats for a strided array ...
:    min, max, mean, sum, std, var, ptp
:   (0, 53, 26.5, 1431, 15.5857..., 242.9166..., 53)
:
: (23)  uniq(ar, return_index=False, return_inverse=False,
:            return_counts=False, axis=0)
:
: (24) is_in(find_in, using, not_in=False)
: (25) n_largest(a, n)... n largest in an array
: (26) n_smallest(a, n).. n smallest counterpart
: (27) rc_vals(a)
: (28) xy_vals(a) ... array to x, y, values
: (29) sort_rows_by_col(a, col=0, descending=False) sort 2d ndarray by column
:     a                           col_sort(a, col=1, descending=False)
:     array([[2, 3, 2, 2],        array([[2, 1, 2, 4],
:            [1, 4, 1, 3],               [2, 3, 2, 2],
:            [2, 1, 2, 4]])              [1, 4, 1, 3]])
: (30) sort_cols_by_row
:
:References:
:-----------
: - https://github.com/numpy/numpy
: - https://github.com/numpy/numpy/blob/master/numpy/lib/_iotools.py
:          striding
: - https://github.com/numpy/numpy/blob/master/numpy/lib/stride_tricks.py
: - http://www.johnvinyard.com/blog/?p=268  for strided arrays
: - https://stackoverflow.com/questions/47469947/
:           as-strided-linking-stepsize-strides-of-conv2d-with-as-strided-
:           strides-paramet#47470711
: - https://stackoverflow.com/questions/48097941/
:           strided-convolution-of-2d-in-numpy  # stride for convolve 4d
: - https://stackoverflow.com/questions/2828059/
:           sorting-arrays-in-numpy-by-column
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
from numpy.lib.stride_tricks import as_strided
from textwrap import dedent, indent

import warnings
warnings.simplefilter('ignore', FutureWarning)

__all__ = ['_func', '_help', '_pad_', 'arr2xyz', 'block', 'block_arr',
           'change_arr', 'doc_func', 'find', 'get_func', 'get_modu',
           'group_pnts', 'group_vals', 'info', 'is_in', 'make_blocks',
           'make_flds', 'n_largest', 'n_smallest', 'nd2struct',
           'num_to_mask', 'num_to_nan', 'rc_vals', 'rec_arr', 'reclass',
           'rolling_stats', 'scale', 'sort_cols_by_row', 'sort_rows_by_col',
           'split_array', 'stride', 'uniq', 'xy_vals']

__xtras__ = ['_check', 'time_deco', 'run_deco', '_demo_tools']
__outside__ = ['dedent', 'indent']

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=2,
                    suppress=True, threshold=550,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script
data_path = script.replace('tools.py', 'Data')


# ---- decorators and helpers ----

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
        dt = t1 - t0
        print("\nTiming function for... {}".format(func.__name__))
        if result is None:
            result = 0
        print("  Time: {: <8.2e}s for {:,} objects".format(dt, result))
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


# ----------------------------------------------------------------------
# (1) doc_func ... code section ...
def doc_func(func=None, verbose=True):
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
    code = "".join(["{:4d}  {}".format(idx+line_num, line)
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
    :Module name... {}
    :
    :----------------------------------------------------------------------
    """
    out = dedent(frmt).format(*args)
    if verbose:
        print(out)
    else:
        return out


# ----------------------------------------------------------------------
# (2) get_func .... code section
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
    :Useage:
    :  import the module containing the function and put the object
    :  name in without quotes... ie
    :  from tools import get_func
    :  get_func(get_func)  # returns this source code etc.
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
    {}\n
    :Source code:
    {}
    :
    :-----------------------------------------------------------------
    """
    import inspect
    from textwrap import dedent, wrap
    lines, ln_num = inspect.getsourcelines(obj)
    if line_nums:
        code = "".join(["{:4d}  {}".format(idx + ln_num, line)
                        for idx, line in enumerate(lines)])
    else:
        code = "".join(["{}".format(line) for line in lines])
    vars_ = ", ".join([i for i in obj.__code__.co_varnames])
    vars_ = wrap(vars_, 50)
    vars_ = "\n".join([i for i in vars_])
    args = [obj.__name__, ln_num, dedent(obj.__doc__), obj.__defaults__,
            obj.__kwdefaults__, indent(vars_, "    "), code]
    code_mem = dedent(frmt).format(*args)
    if verbose:
        print(code_mem)
    else:
        return code_mem


# ----------------------------------------------------------------------
# (3) get_modu .... code section
def get_modu(obj, verbose=True):
    """Get module (script) information, including source code for
    :  documentation purposes.
    :Requires:
    :--------
    :  from textwrap import dedent, indent
    :  import inspect
    :Returns:
    :-------
    :  A string is returned for printing.  It will be the whole module
    :  so use with caution.
    :Useage:
    :------
    :  import tools
    :  tools.get_modu(tools, verbose=True)  # NOTE... no quotes around module
    :
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
    if verbose:
        print(mod_mem)
    else:
        return mod_mem


# ----------------------------------------------------------------------
# (4) info .... code section
def info(a, prn=True):
    """Returns basic information about an numpy array.
    :Requires:
    :--------
    : a - an array
    :Return example:
    :--------------
    :  a = np.arange(2. * 3.).reshape(2, 3) # quick float64 array
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
    : OWNDATA: if 'False', data are a view
    :flags....
    {}
    :array
    :  |__shape {}\n    :  |__ndim  {}\n    :  |__size  {}
    :  |__bytes {}\n    :  |__type  {}\n    :  |__strides  {}
    :dtype      {}
    :  |__kind  {}\n    :  |__char  {}\n    :  |__num   {}
    :  |__type  {}\n    :  |__name  {}\n    :  |__shape {}
    :  |__description
    :  |  |__name, itemsize"""
    dt = a.dtype
    flg = indent(a.flags.__str__(), prefix=':   ')
    info_ = [flg, a.shape, a.ndim, a.size,
             a.nbytes, type(a), a.strides, dt,
             dt.kind, dt.char, dt.num, dt.type, dt.name, dt.shape]
    flds = sorted([[k, v] for k, v in dt.descr])
    out = dedent(frmt).format(*info_) + "\n"
    leader = "".join([":     |__{}\n".format(i) for i in flds])
    leader = leader + ":---------------------"
    out = out + leader
    if prn:
        print(out)
    else:
        return out


# ----------------------------------------------------------------------
# ---- make arrays, change format or arrangement ----
# ----------------------------------------------------------------------
# (5a) num_to_nan ... code section .....
def num_to_nan(a, num=None, copy=True):
    """reverse of nan_to_num introduced in numpy 1.13
    """
    a = a.astype('float64')
    if num is None:
        return a
    if isinstance(num, (list, tuple, np.ndarray)):
        m = is_in(a, num)  # ---- cal to is_in below
        a[m] = np.nan
    else:
        a = np.where(a == num, np.nan, a)
    return a


# (5b) num_to_mask ... code section .....
def num_to_mask(a, num=None, copy=True):
    """reverse of nan_to_num introduced in numpy 1.13
    """
    a = a.astype('float64')
    if num is None:
        return a
    else:
        m = is_in(a, num)  # ---- cal to is_in below
        b = np.ma.MaskedArray(a, mask=m)
    return b


# (6) make_blocks ... code section .....
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


# (7) make_flds .... code section
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
    if names is None:
        names = ", ".join(["{}_{:>02}".format(def_name, i) for i in range(n)])
        dt = easy(f, names=names, defaultfmt=def_name)
    else:
        names = ", ".join(["{}_{:>02}".format(def_name, i) for i in range(n)])
        dt = easy(f, names=names)
    return dt


# (8)  rec_arr .... code section
def rec_arr(a, flds=None, types=None):
    """Sample field creation.  change a uniform array to an array of
    :  mixed dtype
    :Notes:
    :------
    :  The a.T turns the columns to rows so that each row can be assigned a
    :  separate data type.
    :
    :  a = np.arange(20).reshape(4, 5)
    :  a_s = mixed_flds(a, flds='a, b, c, D, e', types='U8, f8, i8, U8, f8')
    :  rec.array([('0',   1.,  2, '3',   4.), ...snip ...
    :             ('15',  16., 17, '18',  19.)],
    :      dtype=[('a', '<U8'), ('b', '<f8'), ('c', '<i8'),
    :             ('D', '<U8'), ('e', '<f8')])
    """
    r, c = a.shape
    if flds is None:
        flds = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:c]
        flds = ", ".join([n for n in flds])
    if types is None:
        types = a.dtype.descr[0][1]
        types = ", ".join(["{}".format(types) for i in range(c)])
    a_s = np.core.records.fromarrays(a.transpose(),
                                     names=flds,
                                     formats=types)
    return a_s


# (9) arr2xyz .... code section
def arr2xyz(a, verbose=False):
    """Produce an array such that the row, column values are used for x,y
    :  and array values for z.
    :See: xy_vals(a) and rc_vals(a) for simpler versions or if you want
    :     structured arrays.
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
    tbl = np.stack((XX.ravel(), YY.ravel(), a.ravel()), axis=1)
    if verbose:
        frmt = """
        ----------------------------
        Meshgrid demo: array to x,y,z table
        :Formulation...
        :  XX,YY = np.meshgrid(np.arange(x.shape[1]),np.arange(x.shape[0]))
        :Input table
        {!r:<}
        :Raveled array, using x.ravel()
        {!r:<}
        :XX in mesh: columns shape[1]
        {!r:<}
        :YY in mesh: rows shape[0]
        {!r:<}
        :Output:
        {!r:<}
        :-----------------------------
        """
        print(dedent(frmt).format(a, a.ravel(), XX, YY, tbl))
    else:
        return tbl


# (10) change_arr ... code section .....
def change_arr(a, order=[], prn=False):
    """Reorder and/or drop columns in an ndarray or structured array.
    :  Fields not included will be dropped in the output array.
    :
    :Reorder fields ----
    : - For a structured/recarray, the desired field order is required.
    : - An ndarray, not using named fields, will require the numerical
    :   order of the fields.
    :
    :Remove fields ----
    : - To remove fields, simply leave them out of the list.  The
    :   order of the remaining fields will be reflected in the output.
    :
    : - This is a convenience function.... see the module header for
    :   one-liner syntax.
    :
    :Tip ----
    : - use... info(a, verbose=True)
    :   This gives field names which can be copied for use here.
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


# (11) nd_struct .... code section
def nd2struct(a, fld_names=None):
    """Return a view of an ndarray as structured array
    :Requires:
    :--------
    : a - ndarray with a uniform dtype.
    : fld_names - a list of strings one for each column/field.
    :     If none are provided, then the field names are assigned
    : from an alphabetical list up to 26 fields
    : - the dtype of the input array is retained, but can be upcast
    :Returns:
    :--------     ---- examples ----
    :  - a = np.arange(2*3).reshape(2,3)
    :  - array([[0, 1, 2],
    :           [3, 4, 5]])  # dtype('int64')
    :  - b = nd2struct(a)
    :    array([(0, 1, 2), (3, 4, 5)],
    :         dtype=[('A', '<i8'), ('B', '<i8'), ('C', '<i8')])
    :  - c = nd2struct(a.astype(np.float64))
    :    array([( 0.,  1.,  2.), ( 3.,  4.,  5.)],
    :         dtype=[('A', '<f8'), ('B', '<f8'), ('C', '<f8')])
    :-------
    """
    if a.dtype.names:  # return if a structured array already
        return a
    alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if a.ndim != 2:
        frmt = "Wrong array shape... read the docs..\n{}"
        print(frmt.format(nd2struct.__doc__))
        return a
    rows, cols = a.shape
    if fld_names is None:
        names = list(alph)[:cols]
    elif (len(fld_names) == cols) and (cols < 26):
        names = fld_names
    else:
        names = ['f{:02.0f}'.format(i) for i in range(cols)]
    return a.view([(n, a.dtype) for n in names]).squeeze(-1)


# (12) scale .... code section
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


# (13) split_array .... code section
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
# stride, block and pad .... code section
#
# (14) _pad_ .... code section .....
def _pad_(a, pad_with=None, size=(1, 1)):
    """To use when padding a strided array for window construction
    : pad_with - selections could be...
    :   ints - 0, +/-128, +/-32768 ie np.iinfo(np.int16).min or max 8, 16, 32
    :   float - 0., np.nan, np.inf, -np.inf or np.finfo(float64).min or max
    """
    print(pad_with)
    if pad_with is None:
        return a
    else:
        new_shape = tuple(i+2 for i in a.shape)
        tmp = np.zeros(new_shape, dtype=a.dtype)
        tmp.fill(pad_with)
        tmp[1:-1, 1:-1] = a
        a = np.copy(tmp, order='C')
        del tmp
    return a


# (15) stride .... code section .....
def stride(a, win=(3, 3), stepby=(1, 1)):
    """Provide a 2D sliding/moving view of an array.
    :  There is no edge correction for outputs. Use a _pad_** function first.
    :
    :Requires:
    :--------
    : as_strided - from numpy.lib.stride_tricks import as_strided
    : a - array or list, usually a 2D array.  Assumes rows is >=1,
    :     it is corrected as is the number of columns.
    : win, stepby - tuple/list/array of window strides by dimensions
    :    1D -    (3,)       (1,)    3 elements, step by 1
    :    2D -    (3, 3)     (1, 1)  3x3 window, step by 1 rows and col.
    :    3D - (1, 3, 3)  (1, 1, 1) 1x3x3, step by 1 row, col, depth
    :Examples:
    :--------
    : - a = np.arange(10)
    :    stride(a, (3,), (1,)) 3 value moving window, step by 1
    :    stride(z, (3,), (2,))
    : array([[0, 1, 2],
    :        [2, 3, 4],
    :        [4, 5, 6],
    :        [6, 7, 8]])
    : - a = np.arange(6*6).reshape(6, 6)
    :    stride(a, (3, 3), (1, 1))  sliding window
    :    stride(a, (3, 3), (3, 3))  block an array
    :
    :Strided variants:
    :----------------
    : a = np.arange(3*5*5).reshape(3, 5, 5)
    : patchify2
    :  x, y   (3, 3)     x, y = patch_shape
    :  n      (3,)       n = a.shape[:-2]
    :  p, q   (5, 5)     p, q = a.shape[-2:]
    :  sn     (100,)     sn = a.strides[:-2]
    :  sp, sq (20, 4)    sp, sq = a.strides[-2:]
    :  l1, l2 (3, 3)     l1, l2 = p - x + 1, q - y + 1
    : out_shp ( n  + ((p - x + 1), (q - y + 1), x, y)
    : out_shp (3,) + ((5 - 3 + 1), (5 - 3 + 1), 3, 3)
    : out_shp (3, 3, 3, 3, 3)         out_shp = n + (l1, l2, x, y)
    : out_stride (100, 20, 4, 20, 4)  out_stride = sn + (sp, sq, sp, sq)
    :
    :Notes:
    :-----
    : - np.product(a.shape) == a.size   # shape product equals array size
    : - To check if the base array and the strided version share memory
    : np.may_share_memory(a, a_s)     # True
    :----------------------------------------------------------
    """
    err = """Array shape, window and/or step size error.
    Use win=(3,3) with stepby=(1,1) for 2D array
    or win=(1,3,3) with stepby=(1,1,1) for 3D
    ----    a.ndim != len(win) != len(stepby) ----
    """
    assert (a.ndim == len(win)) and (len(win) == len(stepby)), err
    shape = np.array(a.shape)  # array shape (r, c) or (d, r, c)
    win_shp = np.array(win)    # window      (3, 3) or (1, 3, 3)
    ss = np.array(stepby)      # step by     (1, 1) or (1, 1, 1)
    newshape = tuple(((shape - win_shp) // ss) + 1) + tuple(win_shp)
    newstrides = tuple(np.array(a.strides) * ss) + a.strides
    a_s = as_strided(a, shape=newshape, strides=newstrides).squeeze()
    return a_s


# (16) block .... code section .....
def block(a, win=(3, 3)):
    """Calls stride with step_by equal to win size.
    :no padding of the array, so this works best when win size is divisible
    : in both directions
    :Note:  see block_arr if you want padding
    """
    a_b = stride(a, win=win, stepby=win)
    return a_b


# (17) block .... code section .....
def block_arr(a, win=[3, 3], nodata=-1, as_masked=False):
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
    if as_masked:
        c = np.ma.masked_equal(c, nodata)
        c.set_fill_value(nodata)
    return c


# ----------------------------------------------------------------------
# ---- querying, working with arrays ----
# ----------------------------------------------------------------------
# (18, 18a) find .... code section
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


# (19)
def group_pnts(a, key_fld='IDs', shp_flds=['Xs', 'Ys']):
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
    uni, idx, inv, cnt = returned
#    from_to = [[idx[i-1], idx[i]] for i in range(1, len(idx))]
    from_to = list(zip(idx, np.cumsum(cnt)))
    subs = [a[shp_flds][i:j] for i, j in from_to]
    groups = [sub.view(dtype='float').reshape(sub.shape[0], -1)
              for sub in subs]
    return groups


# (20)
def group_vals(seq, delta=1, oper='!='):
    """Group consecutive values separated by no more than delta
    :
    : seq - sequence of values
    : delta - difference between consecutive values
    : oper - 'eq', '==', 'ne', '!=', 'gt', '>', 'lt', '<'
    :Reference:
    :---------
    :  https://stackoverflow.com/questions/7352684/
    :    how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
    :    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    """
    valid = ('eq', '==', 'ne', '!=', 'gt', '>', 'lt', '<')
    if oper not in valid:
        raise ValueError("operand not in {}".format(valid))
    elif oper in ('==', 'eq'):
        s = np.split(seq, np.where(np.diff(seq) == delta)[0]+1)
    elif oper in ('!=', 'ne'):
        s = np.split(seq, np.where(np.diff(seq) != delta)[0]+1)
    elif oper in ('>', 'gt'):
        s = np.split(seq, np.where(np.diff(seq) > delta)[0]+1)
    elif oper in ('<', 'lt'):
        s = np.split(seq, np.where(np.diff(seq) < delta)[0]+1)
    else:
        s = seq
    return s


# (21) reclass .... code section
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


# (22) rolling stats .... code section
def rolling_stats(a, no_null=True, prn=True):
    """Statistics on the last two dimensions of an array.
    :Requires
    :--------
    : a - 2D array  **Note, use 'stride' above to obtain rolling stats
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
        a_med = np.median(a, axis=ax)
        a_sum = a.sum(axis=ax)
        a_std = a.std(axis=ax)
        a_var = a.var(axis=ax)
        a_ptp = a_max - a_min
    else:
        a_min = np.nanmin(a, axis=(ax))
        a_max = np.nanmax(a, axis=(ax))
        a_mean = np.nanmean(a, axis=(ax))
        a_med = np.nanmedian(a, axis=(ax))
        a_sum = np.nansum(a, axis=(ax))
        a_std = np.nanstd(a, axis=(ax))
        a_var = np.nanvar(a, axis=(ax))
        a_ptp = a_max - a_min
    if prn:
        s = ['Min', 'Max', 'Mean', 'Med', 'Sum', 'Std', 'Var', 'Range']
        frmt = "...\n{}\n".join([i for i in s])
        v = [a_min, a_max, a_mean, a_med, a_sum, a_std, a_var, a_ptp]
        args = [indent(str(i), '... ') for i in v]
        print(frmt.format(*args))
    else:
        return a_min, a_max, a_mean, a_med, a_sum, a_std, a_var, a_ptp


# (23) uniq  ---- np.unique for versions < 1.13 ----
def uniq(ar, return_index=False, return_inverse=False,
         return_counts=False, axis=None):
    """Taken from, but modified for simple axis 0 and 1 and structured
    :arrays in (N, m) or (N,) format.
    :  https://github.com/numpy/numpy/blob/master/numpy/lib/arraysetops.py
    :To enable determination of unique values in uniform arrays with
    :uniform dtypes.  np.unique in versions < 1.13 need to use this.
    """
    ar = np.asanyarray(ar)
    if np.version.version > '1.13':
        return np.unique(ar, return_index, return_inverse,
                         return_counts, axis=axis)
    if axis is None:
        return np.unique(ar, return_index, return_inverse, return_counts)
    if not (-ar.ndim <= axis < ar.ndim):
        raise ValueError('Invalid axis kwarg specified for unique')

    ar = np.swapaxes(ar, axis, 0)
    orig_shape, orig_dtype = ar.shape, ar.dtype
    # Must reshape to a contiguous 2D array for this to work...
    ar = ar.reshape(orig_shape[0], -1)
    ar = np.ascontiguousarray(ar)

    t_codes = (np.typecodes['AllInteger'] + np.typecodes['Datetime'] + 'S')
    if ar.dtype.char in t_codes:
        dtype = np.dtype((np.void, ar.dtype.itemsize * ar.shape[1]))
    else:
        dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

    try:
        consolidated = ar.view(dtype)
    except TypeError:
        # There's no good way to do this for object arrays, etc...
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype))

    def reshape_uniq(uniq):
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(-1, *orig_shape[1:])
        uniq = np.swapaxes(uniq, 0, axis)
        return uniq

    output = np.unique(consolidated, return_index,
                       return_inverse, return_counts)
    if not (return_index or return_inverse or return_counts):
        return reshape_uniq(output)
    else:
        uniq = reshape_uniq(output[0])
        return (uniq,) + output[1:]


# (24) ---- is_in equivalent to np.isin() for numpy versions < 1.13
def is_in(find_in, using, not_in=False):
    """Equivalent to np.isin for numpy versions < 1.13
    : find_in - the array to check for the elements
    : using - what to use for the check
    :Note:
    : from numpy.lib import NumpyVersion
    : if NumpyVersion(np.__version__) < '1.13.0'):
    """
    find_in = np.asarray(find_in)
    shp = find_in.shape
    using = np.asarray(using)
    uni = False
    inv = False
    if not_in:
        inv = True
    r = np.in1d(find_in, using, assume_unique=uni, invert=inv).reshape(shp)
    return r


# (25) size-based ---- n largest
def n_largest(a, num=1, by_row=True):
    """Return the 'num' largest entries in an array by row sorted by column
    :  Array dimensions <=3 supported
    """
    assert a.ndim <= 3, "Only arrays with ndim <=3 supported"
    if not by_row:
        a = a.T
    num = min(num, a.shape[-1])
    if a.ndim == 1:
        b = np.sort(a)[-num:]
    elif a.ndim >= 2:
        b = np.sort(a)[..., -num:]
    else:
        return None
    return b


# (26) size-based ---- n smallest
def n_smallest(a, num=1, by_row=True):
    """Return the 'n' smallest entries in an array by row sorted by column
    :  Array dimensions <=3 supported
    """
    assert a.ndim <= 3, "Only arrays with ndim <=3 supported"
    if not by_row:
        a = a.T
    num = min(num, a.shape[-1])
    if a.ndim == 1:
        b = np.sort(a)[:num]
    elif a.ndim >= 2:
        b = np.sort(a)[..., :num]
    else:
        return None
    return b


# (27) rc_vals
def rc_vals(a):
    """Convert a 2D ndarray to a structured array with row, col, values array.
    """
    dt = [('r', '<i8'), ('c', '<i8'), ('Val', a.dtype.str)]
    r_c = [(*ij, v) for ij, v in np.ndenumerate(a)]
    vals = np.asarray(r_c, dtype=dt)
    return vals


# (28) xy_vals ----
def xy_vals(a):
    """Convert a 2D ndarray to a structured x, y, vals array.
    """
    r, c = a.shape
    n = r * c
    x, y = np.meshgrid(np.arange(c), np.arange(r))
    dt = [('X', '<i8'), ('Y', '<i8'), ('Vals', a.dtype.str)]
    out = np.zeros((n,), dtype=dt)
    out['X'] = x.ravel()
    out['Y'] = y.ravel()
    out['Vals'] = a.ravel()
    return out


# ---- sorting ---------------------------------------------------------------
# column and row sorting
# (29) sort_rows_by_col ----
def sort_rows_by_col(a, col=0, descending=False):
    """Sort a 2D array by column.
    :  a =array([[0, 1, 2],    array([[6, 7, 8],
    :            [3, 4, 5],           [3, 4, 5],
    :            [6, 7, 8]])          [0, 1, 2]])
    """
    a = np.asarray(a)
    shp = a.shape[0]
    if not (0 <= abs(col) <= shp):
        raise ValueError("column ({}) in range (0 to {})".format(col, shp))
    a_s = a[a[:, col].argsort()]
    if descending:
        a_s = a_s[::-1]
    return a_s


# (30) sort_cols_by_row ----
def sort_cols_by_row(a, col=0, descending=False):
    """Sort the rows of an array in the order of their column values
    :  Uses lexsort """
    a[np.lexsort(np.transpose(a)[::-1])]


# ---- ******* add ... used in nd2struct *****

def pack_last_axis(arr, names=None):
    """
    Then you could do:
    >>> pack_last_axis(uv).tolist()
    to get a list of tuples.
    """
    if arr.dtype.names:
        return arr
    names = names or ['f{}'.format(i) for i in range(arr.shape[-1])]
    return arr.view([(n, arr.dtype) for n in names]).squeeze(-1)


# ----------------------------------------------------------------------
# _help .... code section
def _help():
    """arraytools"""
    _hf = """
    :-------------------------------------------------------------------:
    : ---- arrtools functions  (loaded as 'art') ----
    : ---- from tools.py
    (1)  doc_func(func=None)
         documenting code using inspect
    (2)  get_func(obj, line_nums=True, verbose=True)
         pull in function code
    (3)  get_modu(obj)
         pull in module code
    (4)  info(a)  array info
    (5a, b) num_to_nan, num_to_mask
    (6)  make_blocks(rows=3, cols=3, r=2, c=2, dt='int')
         make arrays consisting of blocks
    (7)  make_flds(n=1, as_type='float', names=None, def_name='col')
         make structured/recarray fields
    (8)  rec_arr(a, flds=None, types=None)
    (9)  arr2xyz(a, verbose=False)
         array (col, rows) to (x, y) and array values for z.
    (10) nd2struct(a)
         convert an ndarray to a structured array with fields
    (11) change(a, order=[], prn=False)
         reorder and/or drop columns
    (12) scale(a, x=2, y=2, num_z=None)
         scale an array up in size by repeating values
    (13) split_array(a, fld='ID')
         split an array using an index field
    (14) _pad_
    (15) stride(a, r_c=(3, 3))
         stride an array for moving window functions
    (16) block
    (17) block_arr(a, win=[3, 3], nodata=-1)
         break an array up into blocks
    (18)  find(a, func, this=None, count=0, keep=[], prn=False, r_lim=2)
         find elements in an array using...
         func - (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
               (      , ==,  !=,  <,   <=,  >,   >=,  >a<, =>a<=,  <a> )
    (19)  group_pnts(a, key_fld='ID', keep_flds=['X', 'Y', 'Z'])
    (20)  group_vals(seq, delta=1, oper='!=')
    (21) reclass(a, bins=[], new_bins=[], mask=False, mask_val=None)
         reclass an array
    (22) rolling_stats((a0, no_null=True, prn=True))
    (23) uniq(ar, return_index=False, return_inverse=False,
              return_counts=False, axis=0)
    (24) is_in
    (25) n_largest(a, num=1, by_row=True)
    (26)    n_smallest(a, num=1, by_row=True)
    (27) rc_vals
    (28) xy_vals
    (29) sort_rows_by_col
    (30)sort_cols_by_row
     ---  _help  this function
    :-------------------------------------------------------------------:
    """
    print(dedent(_hf))


# ----------------------------------------------------------------------
# _demo .... code section
# @run_deco
def _demo_tools():
    """
    : - Run examples of the existing functions.
    """
    a = np.arange(3*4).reshape(3, 4).copy()
    b = nd2struct(a)
    c = np.arange(2*3*4).reshape(2, 3, 4)
    d = np.arange(9*6).reshape(9, 6)
    bloc = block_arr(a, win=[2, 2], nodata=-1)  # for block
    chng = change_arr(b, order=['B', 'C', 'A'], prn=False)
    docf = doc_func(pyramid)
    scal = scale(a, 2)
    a_inf = info(d, prn=False)
    m_blk = make_blocks(rows=3, cols=3, r=2, c=2, dt='int')
    m_fld = str(make_flds(n=3, as_type='int', names=["A", "B", "C"]))
    spl = split_array(b, fld='A')
    stri = stride(a, (3, 3))
    rsta = rolling_stats(d, no_null=True, prn=False)
#    arr = np.load(data_path + '/sample_20.npy')
#    row = arr['County']
#    col = arr['Town']
#    ctab, a0, result, r0, c0 = crosstab(row, col)
#    arr = arr.reshape(arr.shape[0], -1)
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
{}\n
"""
    args = ["-"*62, a, b, c, d, arr2xyz(a), bloc,
            chng.reshape(a.shape[0], -1), docf, scal, a_inf, m_fld, spl, stri,
            m_blk, nd2struct(a), rsta]
    print(frmt.format(*args))
    # del args, d, e


def pyramid(core=9, steps=10, incr=(1, 1), posi=True):
    """Create a pyramid see pyramid_demo.py"""
    a = np.array([core])
    a = np.atleast_2d(a)
    for i in range(1, steps):
        val = core - i
        if posi and (val <= 0):
            val = 0
        a = np.lib.pad(a, incr, "constant", constant_values=(val, val))
    return a


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    _demo_tools()
