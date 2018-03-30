# frmt backup

plus some samples


frmts.py  formatting arrays
===========================

Script:   frmts.py

Author:   Dan.Patterson@carleton.ca

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


  frmt_(a)  example for a =  np.arange(3*4*5).reshape(3, 4, 5)::
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

(1d)redent(lines, spaces=4)
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

>>>  r = "\n".join([f.format(*i) for i in aa])
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

