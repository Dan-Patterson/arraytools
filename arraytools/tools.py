# -*- coding: UTF-8 -*-
"""
arraytools tools
================

Script :   tools.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-03-28

Purpose :  tools for working with numpy arrays

Useage:
-------

>>> import arraytools as art

- `tools.py` and other scripts are part of the arraytools package.
- Access in other programs using .... art.func(params) ....

**Requires**
-------------
  see import section and __init__.py in the `arraytools` folder

**Notes**
---------
**Basic array information**

*np.typecodes*

*np.typecodes.items()*  ... *np.typecodes['AllInteger']*
::
    All: '?bhilqpBHILQPefdgFDGSUVOMm'
    |__AllFloat: 'efdgFDG'
       |__Float: 'efdg'
       |__Complex: 'FDG'
    |__AllInteger: 'bBhHiIlLqQpP'
    |  |__UnsignedInteger: 'BHILQP'
    |  |__Integer: 'bhilqp'
    |__Datetime': 'Mm'
    |__Character': 'c'

*np.sctypes.keys* and *np.sctypes.values*
::
    numpy classes
    |__complex  complex64, complex128, complex256
    |__float    float16, float32, float64, float128
    |__int      int8, int16, int32, int64
    |__uint     uint8, uint16, uint32, uint63
    |__others   bool, object, str, void

**Numbers**
::
   np.inf, -np.inf
   np.iinfo(np.int8).min  or .max = -128, 128
   np.iinfo(np.int16).min or .max = -32768, 32768
   np.iinfo(np.int32) - iinfo(min=-2147483648, max=2147483647, dtype=int32)
   np.finfo(np.float64)
   np.finfo(resolution=1e-15, min=-1.7976931348623157e+308,
            max=1.7976931348623157e+308, dtype=float64)


**Functions**
-------------
    Tool function examples follow...

**1.  doc_func(func=None)** : see get_func and get_modu

**2.  get_func** : retrieve function information
::
    get_func(func, line_nums=True, verbose=True)
    print(art.get_func(art.main))

    Function: .... main ....
    Line number... 1334
    Docs:
    Do nothing
    Defaults: None
    Keyword Defaults: None
    Variable names:
    Source code:
       0  def main():
       1   '''Do nothing'''
       2      pass

**3.  get_modu** : retrieve module info

    get_modu(obj, code=False, verbose=True)

**4.  info(a, prn=True)** : retrieve array information
::
    - array([(0, 1, 2, 3, 4), (5, 6, 7, 8, 9),
             (10, 11, 12, 13, 14), (15, 16, 17, 18, 19)],
      dtype=[('A', '<i8'), ('B', '<i8')... snip ..., ('E', '<i8')])
    ---------------------
    Array information....
    array
      |__shape (4,)
      |__ndim  1
      |__size  4
      |__type  <class 'numpy.ndarray'>
    dtype      [('A', '<i8'), ('B', '<i8') ... , ('E', '<i8')]
      |__kind  V
      |__char  V
      |__num   20
      |__type  <class 'numpy.void'>
      |__name  void320
      |__shape ()
      |__description
         |__name, itemsize
         |__['A', '<i8']
         |__['B', '<i8']
         |__['C', '<i8']
         |__['D', '<i8']
         |__['E', '<i8']

**5.  num_to_nan, num_to_mask** : nan stuff
::
    num_to_nan(a, nums=[2, 3]) .... array([  0.,   1.,  nan,  nan,   4.,   5.])
    num_to_mask(a, nums=[2, 3]) ...
    masked_array(data = [0 1 - - 4 5],
                 mask = [False False  True  True False False],
           fill_value = 999999)

**6.  make_blocks(rows=2, cols=4, r=2, c=2, dt='int')** : create array blocks
::
     array([[0, 0, 1, 1, 2, 2, 3, 3],
            [0, 0, 1, 1, 2, 2, 3, 3],
            [4, 4, 5, 5, 6, 6, 7, 7],
            [4, 4, 5, 5, 6, 6, 7, 7]])

**7.  make_flds(n=1, names=None, default="col")** : example
::
   >>> make_flds(3, names='A,B,C', default='A')
   dtype([('A', '<f8'), ('B', '<f8'), ('C', '<f8')])
   >>> names = 'A, B, C'
   >>> easy(f, names)
   dtype([('A', '<f8'), ('B', '<f8'), ('C', '<f8')])
   >>> names = 'A, B'   # missing a name, so default kicks in
   >>> easy(f, names, name)
   dtype([('A', '<f8'), ('B', '<f8'), ('A00', '<f8')]

**8.  nd_rec and nd_struct** : example

**. nd2struct(a)** : np2rec ... shell around above

ndarray to structured array or recarray

Keep the dtype the same
::
    aa = nd2struct(a)       # produce a structured array from inputs
    aa.reshape(-1,1)   # structured array
    array([[(0, 1, 2, 3, 4)],
           [(5, 6, 7, 8, 9)],
           [(10, 11, 12, 13, 14)],
           [(15, 16, 17, 18, 19)]],
       dtype=[('A', '<i4'), ... snip ... , ('E', '<i4')])

Upcast the dtype
::
       a_f = nd2struct(a.astype('float'))  # note astype allows a view
       array([(0.0, 1.0, 2.0, 3.0, 4.0), ... snip... ,
              (15.0, 16.0, 17.0, 18.0, 19.0)],
          dtype=[('A', '<f8'), ... snip ... , ('E', '<f8')])

**9.  arr2xyz(a, verbose=False)** : convert an array to x,y,z values, using
row/column values for x and y
::
    a= np.arange(2*3).reshape(2,3)
    arr2xyz(a)
    array([[0, 0, 0],
           [1, 0, 1],
           [2, 0, 2],
           [0, 1, 3],
           [1, 1, 4],
           [2, 1, 5]])

**10. change_arr(a, order=[], prn=False)** : merely a convenience function
::
    a = np.arange(4*5).reshape((4, 5))
    change(a, [2, 1, 0, 3, 4])
    array([[ 2,  1,  0,  3,  4],
           [ 7,  6,  5,  8,  9],
           [12, 11, 10, 13, 14],
           [17, 16, 15, 18, 19]])

**shortcuts**
::
    b = a[:, [2, 1, 0, 3, 4]]    # reorder the columns, keeping the rows
    c = a[:, [0, 2, 3]]          # delete columns 1 and 4
    d = a[[0, 1, 3, 4], :]       # delete row 2, keeping the columns
    e = a[[0, 1, 3], [1, 2, 3]]  # keep [0, 1], [1, 2], [3, 3]
                                   => ([ 1, 7, 18])


**12. scale(a, x=2, y=2)** : scale an array by x, y factors
::
      a = np.array([[0, 1, 2], [3, 4, 5]]
      b = scale(a, x=2, y=2)
        =  array([[0, 0, 1, 1, 2, 2],
                  [0, 0, 1, 1, 2, 2],
                  [3, 3, 4, 4, 5, 5],
                  [3, 3, 4, 4, 5, 5]])

using scale with np.tile
::
      art.scale(a, 2,2)         np.tile(art.scale(a, 2, 2), (2, 2))
      array([[0, 0, 1, 1],      array([[0, 0, 1, 1, 0, 0, 1, 1],
             [0, 0, 1, 1],             [0, 0, 1, 1, 0, 0, 1, 1],
             [2, 2, 3, 3],             [2, 2, 3, 3, 2, 2, 3, 3],
             [2, 2, 3, 3]])            [2, 2, 3, 3, 2, 2, 3, 3],
                                       [0, 0, 1, 1, 0, 0, 1, 1],
                                       [0, 0, 1, 1, 0, 0, 1, 1],
                                       [2, 2, 3, 3, 2, 2, 3, 3],
                                       [2, 2, 3, 3, 2, 2, 3, 3]])

**13. split_array(a, fld='Id')**
::
     array 'b'
     array([(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)],
           dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')])
    - split_array(b, fld='A')
    [array([(0, 1, 2, 3)],
          dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')]),
     array([(4, 5, 6, 7)],
          dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')]),
     array([(8, 9, 10, 11)],
          dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')])]

**14. _pad_(a, pad_with=None, size=(1, 1))**


**15. stride(a, r_c=(3, 3))**

Produce a strided array using a window of r_c shape.

Calls _check(a, r_c, subok=False) to check for array compliance
::
      a =np.arange(15).reshape(3,5)
      s = stride(a)    stride     ====>   slide    =====>
      array([[[ 0,  1,  2],  [[ 1,  2,  3],  [[ 2,  3,  4],
              [ 5,  6,  7],   [ 6,  7,  8],   [ 7,  8,  9],
              [10, 11, 12]],  [11, 12, 13]],  [12, 13, 14]]])

`_pad`  to pad an array prior to striding or blocking

`block`  calls stride with non-overlapping blocks with no padding


**16. block(a, win=(3, 3))**


**17.  block_arr(a, win=[3, 3], nodata=-1)**

Block an array given an input array, a window and a nodata value.
::
    a = np.arange(16).reshape(4,4)
    block_arr(a, win=[4, 3], nodata=-1)
    array([[ 0,  1,  2,  3, -1, -1],
           [ 4,  5,  6,  7, -1, -1],
           [ 8,  9, 10, 11, -1, -1],
           [12, 13, 14, 15, -1, -1]]),
    masked_array(data =
        [[[0 1 2]
          [4 5 6]
          [8 9 10]
          [12 13 14]]

         [[3 -- --]
          [7 -- --]
          [11 -- --]
          [15 -- --]]],
    mask .... snipped ....


**18. find(a, func, this=None, count=0, keep=[], prn=False, r_lim=2)**

    func - (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
           (        ==,  !=,  <,   <=,  >,   >=,  >a<, =>a<=,  <a> )

**18a. _func(fn, a, this)**

    called by 'find' see details there
    (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)

Note  see ``find1d_demo.py`` for examples


**19. group_pnts(a, key_fld='ID', keep_flds=['X', 'Y', 'Z'])**

**20. group_vals(seq, stepsize=1)**
::
    seq = [1, 2, 4, 5, 8, 9, 10]
    stepsize = 1
    [array([1, 2]), array([4, 5]), array([ 8,  9, 10])]

**21. reclass(z, bins, new_bins, mask=False, mask_val=None)**

Reclass an array using existing class breaks (bins) and new bins both must be
in ascending order.
::
      z = np.arange(3*5).reshape(3,5)
      bins = [0, 5, 10, 15]
      new_bins = [1, 2, 3, 4]
      z_recl = reclass(z, bins, new_bins, mask=False, mask_val=None)
      ==> .... z                     ==> .... z_recl
      array([[ 0,  1,  2,  3,  4],   array([[1, 1, 1, 1, 1],
             [ 5,  6,  7,  8,  9],          [2, 2, 2, 2, 2],
             [10, 11, 12, 13, 14]])         [3, 3, 3, 3, 3]])


**22. rolling_stats() : stats for a strided array**

    min, max, mean, sum, std, var, ptp


**23. uniq(ar, return_index=False, return_inverse=False, return_counts=False,**
     **axis=0)**

**24. is_in(find_in, using, not_in=False)**

**25. n_largest(a, n)... n largest in an array**

**26. n_smallest(a, n).. n smallest counterpart**

**27. rc_vals(a)**

**28. xy_vals(a) ... array to x, y, values**

**29. sort_rows_by_col(a, col=0, descending=False)**

Sort 2d ndarray by column
::
      a                           col_sort(a, col=1, descending=False)
      array([[2, 3, 2, 2],        array([[2, 1, 2, 4],
             [1, 4, 1, 3],               [2, 3, 2, 2],
             [2, 1, 2, 4]])              [1, 4, 1, 3]])

**30. sort_cols_by_row**

**31. radial_sort(pnts, cent=None)**


References:
----------

general

- https://github.com/numpy/numpy
- https://github.com/numpy/numpy/blob/master/numpy/lib/_iotools.py

striding

- https://github.com/numpy/numpy/blob/master/numpy/lib/stride_tricks.py
- http://www.johnvinyard.com/blog/?p=268

for strided arrays

- https://stackoverflow.com/questions/47469947/as-strided-linking-stepsize-\
strides-of-conv2d-with-as-strided-strides-paramet#47470711
- https://stackoverflow.com/questions/48097941/strided-convolution-of-2d-in-

numpy  # stride for convolve 4d

- https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column

---------------------------------------------------------------------
"""
# ---- imports, formats, constants ----
import sys
from textwrap import dedent, indent
import warnings
import numpy as np
from numpy.lib.stride_tricks import as_strided

warnings.simplefilter('ignore', FutureWarning)

__all__ = ['_func', '_help', '_pad_', 'arr2xyz', 'block', 'block_arr',
           'change_arr', 'doc_func', 'find', 'get_func', 'get_modu',
           'group_pnts', 'group_vals', 'info', 'is_in', 'make_blocks',
           'make_flds', 'n_largest', 'n_smallest', 'nd2struct',
           'num_to_mask', 'num_to_nan', 'rc_vals', 'nd_rec', 'reclass',
           'rolling_stats', 'scale', 'sort_cols_by_row', 'sort_rows_by_col',
           'split_array', 'stride', 'uniq', 'xy_vals']

__xtras__ = ['_check', 'time_deco', 'run_deco', '_demo_tools']
__outside__ = ['dedent', 'indent']

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2,
                    suppress=True, threshold=200,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script
data_path = script.replace('tools.py', 'Data')


# ---- decorators and helpers ----

def time_deco(func):  # timing originally
    """timing decorator function

    - Requires : from functools import wraps

    Uncomment the import or move it to within the script.

    Useage::

        @time_deco  # on the line above the function
        def some_func():
            '''do stuff'''
            return None

    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        t_0 = time.perf_counter()        # start time
        result = func(*args, **kwargs)  # ... run the function ...
        t_1 = time.perf_counter()        # end time
        dt = t_1 - t_0
        print("\nTiming function for... {}".format(func.__name__))
        if result is None:
            result = 0
        print("  Time: {: <8.2e}s for {:,} objects".format(dt, result))
        # return result                   # return the result of the function
        return dt                       # return delta time
    return wrapper


def run_deco(func):
    """Prints basic function information and the results of a run.

    - Requires : from functools import wraps

    Uncomment the import or move it to within the script.

    Useage::

        @func_run  # on the line above the function
        def some_func():
            '''do stuff'''
            return None

    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        """wrapper function"""
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

    Requires:
      import inspect  # module

    Returns
    -------

    A listing of the source code with line numbers

    Parameters
    ----------
    - func : function to document
    - verbose : True prints the result, False returns a string of the result.

    **Notes**::

        Source code for...

        module level
        - inspect.getsourcelines(sys.modules[__name__])[0]

        function level
        - as a list => inspect.getsourcelines(num_41)[0]
        - as a string => inspect.getsource(num_41)

        file level
        - script = sys.argv[0]

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
    if not inspect.isfunction(func):
        out = "\nError... `{}` is not a function, but is of type... {}\n"
        print(out.format(func.__name__, type(func)))
        return None
    if func is None:
        func = demo_func
    script = sys.argv[0]  # a useful way to get a file's name
    lines, line_num = inspect.getsourcelines(func)
    code = "".join(["{:4d}  {}".format(idx+line_num, line)
                    for idx, line in enumerate(lines)])
    nmes = ['args', 'varargs', 'varkw', 'defaults', 'kwonlyargs',
            'kwonlydefaults', 'annotations']
    f = inspect.getfullargspec(func)
    f_args = "\n".join([str(i) for i in list(zip(nmes, list(f)))])
    args = [line_num, code,
            inspect.getcomments(func),
            inspect.isfunction(func),
            inspect.ismethod(func),
            inspect.getmodulename(script),
            f_args]
    frmt = """
    :----------------------------------------------------------------------
    :---- doc_func(func) ----
    :Code for a function on line...{}...
    :
    {}
    Comments preceeding function
    {}
    function?... {} ... or method? {}
    Module name... {}
    Full specs....
    {}
    ----------------------------------------------------------------------
    """
    out = (dedent(frmt)).format(*args)
    if verbose:
        print(out)
    else:
        return out


# ----------------------------------------------------------------------
# (2) get_func .... code section
def get_func(func, line_nums=True, verbose=True):
    """Get function information (ie. for a def)

    Requires
    --------

    - from textwrap import dedent, indent, wrap
    - import inspect

    Returns
    -------

    The function information includes arguments and source code.
    A string is returned for printing.

    Notes
    -----

    Import the module containing the function and put the object name in
    without quotes...

        from tools import get_func

        get_func(get_func)  # returns this source code etc.

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

    if not inspect.isfunction(func):
        out = "\nError... `{}` is not a function, but is of type... {}\n"
        print(out.format(func.__name__, type(func)))
        return None

    lines, ln_num = inspect.getsourcelines(func)
    if line_nums:
        code = "".join(["{:4d}  {}".format(idx + ln_num, line)
                        for idx, line in enumerate(lines)])
    else:
        code = "".join(["{}".format(line) for line in lines])

    vars_ = ", ".join([i for i in func.__code__.co_varnames])
    vars_ = wrap(vars_, 50)
    vars_ = "\n".join([i for i in vars_])
    args = [func.__name__, ln_num, dedent(func.__doc__), func.__defaults__,
            func.__kwdefaults__, indent(vars_, "    "), code]
    code_mem = dedent(frmt).format(*args)
    if verbose:
        print(code_mem)
    else:
        return code_mem


# ----------------------------------------------------------------------
# (3) get_modu .... code section
def get_modu(obj, code=False, verbose=True):
    """Get module (script) information, including source code for
    documentation purposes.

    Requires
    --------
    >>> from textwrap import dedent, indent
    >>> import inspect

    Returns
    -------
    A string is returned for printing.  It will be the whole module
    so use with caution.

    Notes
    -----
    Useage::

    >>> import tools
    >>> tools.get_modu(tools, code=False, verbose=True)
    >>> # No quotes around module name, code=True for module code

   """
    frmt = """
    :-----------------------------------------------------------------
    :Module: .... {} ....
    :------
    :File: ......
    {}\n
    :Docs: ......
    {}\n
    :Members: .....
    {}
    """
    frmt0 = """
    :{}
    :-----------------------------------------------------------------
    """
    frmt1 = """
    :Source code: .....
    {}
    :
    :-----------------------------------------------------------------
    """
    import inspect
    from textwrap import dedent

    if not inspect.ismodule(obj):
        out = "\nError... `{}` is not a module, but is of type... {}\n"
        print(out.format(obj.__name__, type(obj)))
        return None
    if code:
        lines, _ = inspect.getsourcelines(obj)
        frmt = frmt + frmt1
        code = "".join(["{:4d}  {}".format(idx, line)
                        for idx, line in enumerate(lines)])
    else:
        lines = code = ""
        frmt = frmt + frmt0
    memb = [i[0] for i in inspect.getmembers(obj)]
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

    Requires:
    --------

    - a : an array
    - prn : True to print, False to return as string.

    Returns
    -------

    example::

        a = np.arange(2. * 3.).reshape(2, 3) # quick float64 array
        arr_info(a)
        ---------------------
        Array information....
         OWNDATA: if 'False', data are a view
        flags....
        ... snip ...
        array
            |__shape (2, 3)
            |__ndim  2
            |__size  6
            |__bytes
            |__type  <class 'numpy.ndarray'>
            |__strides  (24, 8)
        dtype      float64
            |__kind  f
            |__char  d
            |__num   12
            |__type  <class 'numpy.float64'>
            |__name  float64
            |__shape ()
            |__description
                 |__name, itemsize
                 |__['', '<f8']
    ---------------------
    """
    if not isinstance(a, (np.ndarray, np.ma.core.MaskedArray)):
        s = "\n... Requires a numpy ndarray or variant...\n... Read the docs\n"
        print(s)
        return None
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
def num_to_nan(a, nums=None):
    """Reverse of nan_to_num introduced in numpy 1.13

    Example
    -------
    >>> a = np.arange(10)
    >>> num_to_nan(a, num=[2, 3])
    array([  0.,   1.,   nan,  nan,   4.,   5.,   6.,   7.,   8.,   9.])
    """
    a = a.astype('float64')
    if nums is None:
        return a
    if isinstance(nums, (list, tuple, np.ndarray)):
        m = is_in(a, nums)  # ---- call to is_in below
        a[m] = np.nan
    else:
        a = np.where(a == nums, np.nan, a)
    return a


# (5b) num_to_mask ... code section .....
def num_to_mask(a, nums=None, hardmask=True):
    """Reverse of nan_to_num introduced in numpy 1.13

    Example
    -------
    >>> a = np.arange(10)
    >>> art.num_to_mask(a, nums=[1, 2, 4])
    masked_array(data = [0 - - 3 - 5 6 7 8 9],
                mask = [False  True  True False  True False
                        False False False False], fill_value = 999999)
    """
    if nums is None:
        return a
    else:
        m = is_in(a, nums)  # ---- call to is_in below
        b = np.ma.MaskedArray(a, mask=m, hard_mask=hardmask)
    return b


# (6) make_blocks ... code section .....
#
def make_blocks(rows=3, cols=3, r=2, c=2, dt='int'):
    """Make a block array with rows * cols containing r*c sub windows.
    Specify the rows, columns, then the block size as r, c and dtype
    Use `scale`, if you want specific values during array construction.

    Requires
    --------
    - rows : rows in initial array
    - cols : columns in the initial array
    - r : rows in sub window
    - c : columns in sub window
    - dt : array data type

    Returns
    --------

    The defaults produce an 8 column by 8 row array numbered from
    0 to (rows*cols) - 1

    array.shape = (rows * r, cols * c) = (6, 6)

    >>> make_blocks(rows=3, cols=3, r=2, c=2, dt='int')
    array([[0, 0, 1, 1, 2, 2],
           [0, 0, 1, 1, 2, 2],
           [3, 3, 4, 4, 5, 5],
           [3, 3, 4, 4, 5, 5],
           [6, 6, 7, 7, 8, 8],
           [6, 6, 7, 7, 8, 8]])

    """
    a = np.arange(rows*cols, dtype=dt).reshape(rows, cols)
    a = scale(a, x=r, y=c)
    return a


# (7) make_flds .... code section
#
def make_flds(n=1, as_type='float', names=None, def_name="col"):
    """Create float or integer fields for statistics and their names.

    Requires
    --------
        n : number of fields to create excluding the names field

        def_name : base name to use, numeric values will be produced for each
                   dimension for the 3D array, ie Values_00... Values_nn

    Returns
    -------

    - a dtype : which contains the necessary fields to contain the values.

    >>> from numpy.lib._iotools import easy_dtype as easy
    >>> make_flds(n=1, as_type='float', names=None, def_name="col")
    dtype([('col_00', '<f8')])

    >>> make_flds(n=2, as_type='int', names=['f01', 'f02'], def_name="col")
    dtype([('f01', '<i8'), ('f02', '<i8')])

    Don't forget the above, a cool way to create fields quickly

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
        dt = easy(f, names=names)
    return dt


# (8)  make nd_rec, nd_struct.... code section
#
def nd_rec(a, flds=None, types=None):
    """Change a uniform array to an array of mixed dtype as a recarray

    Requires:
    ---------
    flds : string or None
        flds='a, b, c'
    types : string or None
        types='U8, f8, i8'

    See also:
    ---------
    nd_struct : alternate using lists rather than string inputs

    Notes:
    -----
    The a.T turns the columns to rows so that each row can be assigned a
    separate data type.

    Example::

       a = np.arange(9).reshape(3, 3)
       a_r = nd_rec(a, flds='a, b, c', types='U8, f8, i8')
       a_r
       rec.array([('0',  1., 2), ('3',  4., 5), ('6',  7., 8)],
          dtype=[('a', '<U8'), ('b', '<f8'), ('c', '<i8')])

    """
    _, c = a.shape
    if flds is None:
        flds = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:c]
        flds = ", ".join([n for n in flds])
    if types is None:
        types = a.dtype.str  # a.dtype.descr[0][1]
        types = ", ".join(["{}".format(types) for i in range(c)])
    a_r = np.core.records.fromarrays(a.transpose(),
                                     names=flds,
                                     formats=types)
    return a_r


def nd_struct(a, flds=None, types=None):
    """"Change an array with uniform dtype to an array of mixed dtype as a
    structured array.

    Requires:
    ---------
    flds : list or None
        flds=['A', 'B', 'C']
    types : list or None
        types=['U8', 'f8', 'i8']

    See also:
    ---------
    nd_rec : alternate using strings rather than list inputs

    Example::

        a = np.arange(9).reshape(3, 3)
        a_s = nd_struct(a, flds=['A', 'B', 'C'], types=['U8', 'f8', 'i8'])
        a_s
        array([('0',  1., 2), ('3',  4., 5), ('6',  7., 8)],
              dtype=[('A', '<U8'), ('B', '<f8'), ('C', '<i8')])

    Timing of nd_rec and nd_struct

    >>> %timeit nd_rec(a, flds='a, b, c', types='U8, f8, i8')
    465 µs ± 53 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    >>> %timeit nd_struct(a, flds=['A', 'B', 'C'], types=['U8', 'f8', 'i8'])
    253 µs ± 27.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    """
    _, c = a.shape
    dt_base = [a.dtype.str] * c  # a.dtype.descr[0][1]
    if flds is None:
        flds = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:c]
    if types is None:
        types = dt_base
    dt0 = np.dtype(list(zip(flds, dt_base)))
    dt1 = list(zip(flds, types))
    a_s = a.view(dtype=dt0).squeeze(axis=-1).astype(dt1)
    return a_s


#  nd_struct and np2rec .... code section
def nd2struct(a, fld_names=None):
    """Return a view of an ndarray as structured array with a uniform dtype/

    Parameters
    ----------
        a : ndarray with a uniform dtype.

        fld_names : a list of strings one for each column/field.

    If none are provided, then the field names are assigned
    from an alphabetical list up to 26 fields
    The dtype of the input array is retained, but can be upcast.

    Examples
    --------
        >>> a = np.arange(2*3).reshape(2,3)
        array([[0, 1, 2],
               [3, 4, 5]])  # dtype('int64')
        >>> b = nd2struct(a)
        array([(0, 1, 2), (3, 4, 5)],
              dtype=[('A', '<i8'), ('B', '<i8'), ('C', '<i8')])
        >>> c = nd2struct(a.astype(np.float64))
        array([( 0.,  1.,  2.), ( 3.,  4.,  5.)],
              dtype=[('A', '<f8'), ('B', '<f8'), ('C', '<f8')])

    See Also
    --------
    pack_last_axis(arr, names=None) at the end

    :-----------------------------------------------------------
    """
    if a.dtype.names:  # return if a structured array already
        return a
    alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if a.ndim != 2:
        frmt = "Wrong array shape... read the docs..\n{}"
        print(frmt.format(nd2struct.__doc__))
        return a
    _, cols = a.shape
    if fld_names is None:
        names = list(alph)[:cols]
    elif (len(fld_names) == cols) and (cols < 26):
        names = fld_names
    else:  # from... pack_last_axis
        names = ['f{:02.0f}'.format(i) for i in range(cols)]
    return a.view([(n, a.dtype) for n in names]).squeeze(-1)


def nd2rec(a, fld_names=None):
    """Shell to nd2struct but yielding a recarray.
    """
    a = nd2struct(a, fld_names=None)
    return a.view(type=np.recarray)


# (9) arr2xyz .... code section
#
def arr2xyz(a, keep_masked=False, verbose=False):
    """Produce an array such that the row, column values are used for x,y
    and array values for z.  Masked arrays are sorted

    Returns
    --------
    A mesh grid with values, dimensions and shapes are changed so
    that ndim=2, ie shape(3,4,5), ndim=3 becomes shape(12,5), ndim=2

    Example::

        >>> a = np.arange(9).reshape(3, 3)
        array([[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]])
        >>> arr2xyz(am, keep_masked=True)   # keep the masked values...
        masked_array(data =
        [[0 0 0]
         [1 0 -]
         [2 0 2]
         [0 1 -]
         [1 1 4]
         [2 1 -]
         [0 2 6]
         [1 2 7]
         [2 2 8]],
                 mask =
         [[False False False]... snip
         [False False False]],
               fill_value = 999999)

    >>> arr2xyz(am, keep_masked=False)  # remove the masked values
    array([[0, 0, 0],
           [2, 0, 2],
           [1, 1, 4],
           [0, 2, 6],
           [1, 2, 7],
           [2, 2, 8]])


    See also
    --------
        `xy_vals(a)` and

        `rc_vals(a)`

        for simpler versions or if you want structured arrays.

        `num_to_mask(a)` and  `num_to_nan(a)` to produce masks prior to
        conversion

    """
    if a.ndim == 1:
        a = a.reshape(a.shape[0], 1)
    if a.ndim > 2:
        a = a.reshape(np.product(a.shape[:-1]), a.shape[-1])
    r, c = a.shape
    XX, YY = np.meshgrid(np.arange(c), np.arange(r))
    XX = XX.ravel()
    YY = YY.ravel()
    if isinstance(np.ma.getmask(a), np.ndarray):
        tbl = np.ma.vstack((XX, YY, a.ravel()))
        tbl = tbl.T
        if not keep_masked:
            m = tbl[:, 2].mask
            tbl = tbl[~m].data
    else:
        tbl = np.stack((XX, YY, a), axis=1)
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

# (27) rc_vals
def rc_vals(a):
    """Convert a 2D ndarray to a structured row, col, values array.
    """
    dt = [('r', '<i8'), ('c', '<i8'), ('Val', a.dtype.str)]
    r_c = [(*ij, v) for ij, v in np.ndenumerate(a)]
    vals = np.asarray(r_c, dtype=dt)
    return vals


# (28) xy_vals ----
def xy_vals(a):
    """Convert a 2D ndarray to a structured x, y, values array.
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

# ----------------------------------------------------------------------------
# (10) change_arr ... code section .....
#
def change_arr(a, order=None, prn=False):
    """Reorder and/or drop columns in an ndarray or structured array.

    Fields not included will be dropped in the output array.

    Parameters
    ----------
    order : list of fields
        fields in the order that you want them

    To reorder fields : ['a', 'c', 'b']
        For a structured/recarray, the desired field order is required.
        An ndarray, not using named fields, will require the numerical
        order of the fields.

    To remove fields : ['a', 'c']  # `b` dropped
        To remove fields, simply leave them out of the list.  The
        order of the remaining fields will be reflected in the output.
        This is a convenience function.... see the module header for
        one-liner syntax.

    Tip
        Use... `info(a, verbose=True)`
        This gives field names which can be copied for use here.

    """
    if order is None or (not isinstance(order, (list, tuple))):
        print("Order not given in a list or tuple")
        return a
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


# (12) scale .... code section
def scale(a, x=2, y=2, num_z=None):
    """Scale the input array repeating the array values up by the
    x and y factors.

    Parameters:
    ----------
    `a` : An ndarray, 1D arrays will be upcast to 2D.

    `x y` : Factors to scale the array in x (col) and y (row).  Scale factors
    must be greater than 2.

    `num_z` : For 3D, produces the 3rd dimension, ie. if num_z = 3 with the
    defaults, you will get an array with shape=(3, 6, 6),

    Examples:
    --------
    >>> a = np.array([[0, 1, 2], [3, 4, 5]]
    >>> b = scale(a, x=2, y=2)
    array([[0, 0, 1, 1, 2, 2],
           [0, 0, 1, 1, 2, 2],
           [3, 3, 4, 4, 5, 5],
           [3, 3, 4, 4, 5, 5]])

    Notes:
    -----
    >>> a = np.arange(2*2).reshape(2,2)
    array([[0, 1],
           [2, 3]])

    >>> frmt_(scale(a, x=2, y=2, num_z=2))
    Array... shape (3, 4, 4), ndim 3, not masked
      0, 0, 1, 1    0, 0, 1, 1    0, 0, 1, 1
      0, 0, 1, 1    0, 0, 1, 1    0, 0, 1, 1
      2, 2, 3, 3    2, 2, 3, 3    2, 2, 3, 3
      2, 2, 3, 3    2, 2, 3, 3    2, 2, 3, 3
      sub (0)       sub (1)       sub (2)

    """
    if (x < 1) or (y < 1):
        print("x or y scale < 1... read the docs\n{}".format(scale.__doc__))
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
    `fld` field.  It is assumed that there is a sequential ordering to
    the values in the field.  If there is not, use np.where in conjunction
    with np.unique or sort the array first.

    Parameters
    ----------
    `a` : A structured or recarray.

    `fld` : A numeric field assumed to be sorted which indicates which group
    a record belongs to.

    Returns
    -------
    A list of arrays split on the categorizing field

    """
    return np.split(a, np.where(np.diff(a[fld]))[0] + 1)


# ----------------------------------------------------------------------
# stride, block and pad .... code section
#
# (14) _pad_ .... code section .....
def _pad_(a, pad_with=None, size=(1, 1)):
    """To use when padding a strided array for window construction.

    Parameters:
    ----------
    pad_with : Selections could be.
        ints - 0, +/-128, +/-32768 `np.iinfo(np.int16).min or max 8, 16, 32`.

        float - 0., np.nan, np.inf, `-np.inf` or `np.finfo(float64).min or max`

    size :
        Size of padding on sides as rows and columns.
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
    There is no edge correction for outputs. Use a _pad_** function first.

    Requires
    --------

    `as_strided` - from numpy.lib.stride_tricks import as_strided

    `a` - array or list, usually a 2D array.  Assumes rows is >=1,
    it is corrected as is the number of columns.

    `win, stepby` - tuple/list/array of window strides by dimensions
    ::
        - 1D - (3,)       (1,)       3 elements, step by 1
        - 2D - (3, 3)     (1, 1)     3x3 window, step by 1 rows and col.
        - 3D - (1, 3, 3)  (1, 1, 1)  1x3x3, step by 1 row, col, depth

    Examples
    --------
    >>> a = np.arange(10)
    >>> stride(a, (3,), (1,)) 3 value moving window, step by 1
    >>> stride(z, (3,), (2,))
    array([[0, 1, 2],
           [2, 3, 4],
           [4, 5, 6],
           [6, 7, 8]])
    >>> a = np.arange(6*6).reshape(6, 6)
    #    stride(a, (3, 3), (1, 1))  sliding window
    #    stride(a, (3, 3), (3, 3))  block an array

    Notes:
    -----
    - np.product(a.shape) == a.size   # shape product equals array size
    - To check if the base array and the strided version share memory
    - np.may_share_memory(a, a_s)     # True

    ----------------------------------------------------------
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


def sliding_window_view(x, shape=None):
    """Create rolling window views of the 2D array with the given shape.
    proposed for upcoming numpy version.
    """
    if shape is None:
        shape = x.shape
    o = np.array(x.shape) - np.array(shape) + 1  # output shape
    strides = x.strides
    view_shape = np.concatenate((o, shape), axis=0)
    view_strides = np.concatenate((strides, strides), axis=0)
    return np.lib.stride_tricks.as_strided(x, view_shape, view_strides)


# (16) block .... code section .....
def block(a, win=(3, 3)):
    """Calls stride with step_by equal to win size.
    No padding of the array, so this works best when win size is divisible
    in both directions

    Note:
        see block_arr if you want padding
    """
    a_b = stride(a, win=win, stepby=win)
    return a_b


# (17) block .... code section .....
def block_arr(a, win=[3, 3], nodata=-1, as_masked=False):
    """Block array into window sized chunks padding to the right and bottom
    to accommodate array and window shape.

    Parameters
    ----------
        `a` - 2D array

        `win` - [rows, cols], aka y,x, m,n sized window

        `nodata - to use for the mask

    Returns
    -------
        The padded array and the masked array blocked.

    Reference
    ---------
        `http://stackoverflow.com/questions/40275876/
             how-to-reshape-this-image-array-in-python`

    extras::

        def block_2(a, blocks=2)
            B = blocks # Blocksize
            m, n = a.shape
            out = a.reshape(m//B, B, n//B, B).swapaxes(1, 2).reshape(-1, B, B)
            return out

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
    """Called by 'find' see details there
    (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
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
def find(a, func, this=None, count=0, keep=None, prn=False, r_lim=2):
    """Find the conditions that are met in an array, defined by `func`.
    `this` is the condition being looked for.  The other parameters are defined
    in the Parameters section.

        >>> a = np.arange(10)
        >>> find(a, 'gt', this=5)
        array([6, 7, 8, 9])

    Parameters
    ----------
    `a` :
        Array or array like.
    `func` :
        `(cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)`
        (        ==,  !=,  <,   <=,  >,   >=,  >a<, =>a<=,  <a> )
    `count` :
        only used for recursive functions
    `keep` :
        for future use
    `verbose` :
        True for test printing
    `max_depth` :
        prevent recursive functions running wild, it can be varied

    Recursive functions:
    -------------------
    cumsum :
        An example of using recursion to split a list/array of data
        parsing the results into groups that sum to this.  For example,
        split input into groups where the total population is less than
        a threshold (this).  The default is to use a sequential list,
        however, the inputs could be randomized prior to running.

    Returns
    -------
        A 1D or 2D array meeting the conditions

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
        elif count == r_lim:
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
    `arcpy.da.FeatureClassToNumPyArray`.

    Parameters:
    ---------
    `a`
        a structured array, assuming ID, X, Y, {Z} and whatever else
        the array is assumed to be sorted... which will be the case
    `key_fld`
        Normally this is the `IDs` or similar
    `shp_flds`
        The fields that are used to produce the geometry.

    Returns:
    -------
        See np.unique descriptions below

    References:
    ----------
        https://jakevdp.github.io/blog/2017/03/22/group-by-from-scratch/

        http://esantorella.com/2016/06/16/groupby/

    Notes:
    -----
        split-apply-combine .... that is the general rule

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

    Parameters
    ----------
    `seq` :
        sequence of values
    `delta` :
        difference between consecutive values
    `oper` :
        'eq', '==', 'ne', '!=', 'gt', '>', 'lt', '<'

    Reference
    ---------
        `https://stackoverflow.com/questions/7352684/
         how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy`

    Notes
    -----
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
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
def reclass(a, bins=None, new_bins=[], mask_=False, mask_val=None):
    """Reclass an array of integer or floating point values.

    Requires:
    --------
    bins :
        sequential list/array of the lower limits of each class
        include one value higher to cover the upper range.
    new_bins :
        new class values for each bin
    mask :
        whether the raster contains nodata values or values to
        be masked with mask_val

    Array dimensions will be squeezed.

    Example
    -------
    inputs::

        z = np.arange(3*5).reshape(3,5)
        bins = [0, 5, 10, 15]
        new_bins = [1, 2, 3, 4]
        z_recl = reclass(z, bins, new_bins, mask=False, mask_val=None)

    outputs::

        ==> .... z                     ==> .... z_recl
        array([[ 0,  1,  2,  3,  4],   array([[1, 1, 1, 1, 1],
               [ 5,  6,  7,  8,  9],          [2, 2, 2, 2, 2],
               [10, 11, 12, 13, 14]])         [3, 3, 3, 3, 3]])

    """
    a_rc = np.zeros_like(a)
    c_0 = isinstance(bins, (list, tuple))
    c_1 = isinstance(new_bins, (list, tuple))
    err = "Bins = {} new = {} won't work".format(bins, new_bins)
    if not c_0 or not c_1:
        print(err)
        return a
    if len(bins) < 2:  # or (len(new_bins <2)):
        print(err)
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

    Requires
    --------
    a :
        2D array  **Note, use 'stride' above to obtain rolling stats
    no_null :
        boolean, whether to use masked values (nan) or not.
    prn :
        boolean, to print the results or return the values.

    Returns
    -------
        The results return an array of 4 dimensions representing the original
        array size and block size

        eg. original = 6x6 array   block = 3x3
            breaking the array into 4 chunks
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
    arrays in (N, m) or (N,) format.

    To enable determination of unique values in uniform arrays with
    uniform dtypes.  np.unique in versions < 1.13 need to use this.

    https://github.com/numpy/numpy/blob/master/numpy/lib/arraysetops.py
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
        """reshape uniq"""
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

    Parameters
    ----------

    find_in :
        the array to check for the elements
    using :
        what to use for the check

    Note
    ----

    >>> from numpy.lib import NumpyVersion
    >>> if NumpyVersion(np.__version__) < '1.13.0'):
        # can add for older versions later
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
    """Return the 'num' largest entries in an array by row sorted by column.
    Array dimensions <=3 supported
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
    """Return the 'n' smallest entries in an array by row sorted by column.
    Array dimensions <=3 supported
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


# ---- sorting ---------------------------------------------------------------
# column and row sorting
# (29) sort_rows_by_col ----
def sort_rows_by_col(a, col=0, descending=False):
    """Sort a 2D array by column.

    >>> a =array([[0, 1, 2],    array([[6, 7, 8],
                  [3, 4, 5],           [3, 4, 5],
                  [6, 7, 8]])          [0, 1, 2]])
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
    return a[np.lexsort(np.transpose(a)[::-1])]


# (31) radial sort -----
def radial_sort(pnts, cent=None):
    """Sort about the point cloud center or from a given point

    Requires:
    ---------

    pnts :
        an array of points (x,y) as array or list
    cent :
        list, tuple, array of the center's x,y coordinates
        cent = [0, 0] or np.array([0, 0])

    Returns:
    --------
        The angles in the range -180, 180 x-axis oriented
    """
    pnts = np.asarray(pnts, dtype='float64')
    if cent is None:
        cent = pnts.mean(axis=0)
    ba = pnts - cent
    ang_ab = np.arctan2(ba[:, 1], ba[:, 0])
    ang_ab = np.degrees(ang_ab)
    sort_order = np.argsort(ang_ab)
    return ang_ab, sort_order
# ---- ******* add ... used in nd2struct *****


def pack_last_axis(arr, names=None):
    """Find source *****
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
    """arraytools.tools help...

    Function list follows:
    """
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
    (10) change(a, order=[], prn=False)
         reorder and/or drop columns
    (11) nd2struct(a)
         convert an ndarray to a structured array with fields
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
    (26) n_smallest(a, num=1, by_row=True)
    (27) rc_vals
    (28) xy_vals
    (29) sort_rows_by_col
    (30) sort_cols_by_row
    (31) radial_sort
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
:(4) doc_func(col_hdr) ... documenting a function...
{}\n
:(5) scale() ... scale an array up by an integer factor...
{}\n
:(6) array info ... info(a)
{}\n
:(7) make_flds() ... create default field names ...
{}\n
:(8) split_array() ... split an array according to an index field
{}\n
:(9) stride() ... stride an array ....
{}\n
:(10) make_blocks(rows=3, cols=3, r=2, c=2, dt='int')
{}\n
:(11) nd_struct() ... make a structured array from another array ...
{!r:}\n
:(12) rolling_stats()... stats for a strided array ...
:    min, max, mean, sum, std, var, ptp
{}\n
"""
    args = ["-"*62, a, b, c, d,
            arr2xyz(a), bloc, chng.reshape(a.shape[0], -1), docf, scal,  # 1 -5
            a_inf, m_fld, spl, stri, m_blk, nd2struct(a), rsta]  # 6- 12
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
