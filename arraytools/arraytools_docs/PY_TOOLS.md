PY_TOOLS
=========

Sample output of some of the functions.

-----
**folders(path, first=True, prefix="")**

Recursively calls itself just pulling out the folder names with formatting.

```
folders('C:/Git_Dan'
------------------------------
|.... Folder listing for ....|
|--C:/Git_Dan
|--C:\Git_Dan\arraytools
  -                     \.spyproject
  -                     \analysis
  -                              \__pycache__
  -                     \Data
  -                          \numpy_demos.gdb
  -                          \terrain
  -                     \geomtools
  -                               \__pycache__
  -                     \graphing
  -                              \__pycache__
  -                     \rasters
  -                             \Data
  -                             \__pycache__
  -                     \stats
  -                           \Extras
  -                           \__pycache__
  -                     \tifffile
  -                              \__pycache__
  -                     \__pycache__
|--C:\Git_Dan\a_Data
  -                 \arcgis_testing.gdb
  -                 \arcpytools_demo.gdb
  -                 \ImportLog
  -                 \Index
  -                       \arcpytools_demo
  -                       \arcpytools_demo2
  -                       \Thumbnail
  -                 \sample_data.gdb
  -                 \testdata.gdb
|--C:\Git_Dan\__pycache__
```

-----

**dirr(obj, colwise=False, cols=4, sub=None, prn=True)**

Return a `dir` on an object and return it in a more structured manner.
You can specify the number of columns, get numbering and some limited subsampling.

```
a = []  # just a list

dirr(a, True, 5)

----------------------------------------------------------------------
| dir(<class 'list'>) ...
|    np version
-------
  (001)    __add__           __getattribute__  __le__            __reversed__      copy              
  (006)    __class__         __getitem__       __len__           __rmul__          count             
  (011)    __contains__      __gt__            __lt__            __setattr__       extend            
  (016)    __delattr__       __hash__          __mul__           __setitem__       index             
  (021)    __delitem__       __iadd__          __ne__            __sizeof__        insert            
  (026)    __dir__           __imul__          __new__           __str__           pop               
  (031)    __doc__           __init__          __reduce__        __subclasshook__  remove            
  (036)    __eq__            __init_subclass__ __reduce_ex__     append            reverse           
  (041)    __format__        __iter__          __repr__          clear             sort              
  (046)    __ge__                                                                                    
```

Change the row/column option.

```
dirr(a, False, 5)

----------------------------------------------------------------------
| dir(<class 'list'>) ...
|    np version
-------
  (001)    __add__           __class__         __contains__      __delattr__       __delitem__       
  (006)    __dir__           __doc__           __eq__            __format__        __ge__            
  (011)    __getattribute__  __getitem__       __gt__            __hash__          __iadd__          
  (016)    __imul__          __init__          __init_subclass__ __iter__          __le__            
  (021)    __len__           __lt__            __mul__           __ne__            __new__           
  (026)    __reduce__        __reduce_ex__     __repr__          __reversed__      __rmul__          
  (031)    __setattr__       __setitem__       __sizeof__        __str__                             
  (036)    __subclasshook__  append            clear             copy                                
  (041)    count             extend            index             insert                              
  (046)    pop               remove            reverse           sort                                
```

Subsample out those that begin with a `c`.

```
dirr(a, False, 3, 'c*')

----------------------------------------------------------------------
| dir(<class 'list'>) ...
|    np version
-------
  (001)    clear copy  count 
```

