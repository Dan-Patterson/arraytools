# -*- coding: UTF-8 -*-
"""
a_io.py
=======

Script :   a_io.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-04-22

Purpose : Basic io tools for numpy arrays and arcpy

Notes :
::
    1.  load_npy    - load numpy npy files
    2.  save_npy    - save array to *.npy format
    3.  read_txt    - read array created by save_txtt
    4.  save_txt    - save array to npy format
    5.  arr_json    - save to json format
    6.  array2raster - save array to raster
    7.  rasters2nparray - batch rasters to numpy array

---------------------------------------------------------------------
"""
# ---- imports, formats, constants ----
import sys
import os
import numpy as np


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['load_npy', 'save_npy',
           'read_txt', 'save_txt',
           'arr_json',
           'array2raster', 'rasters2nparray',
           ]


# ----------------------------------------------------------------------
# (1) load_npy .... code section ---
def load_npy(f_name, all_info=False):
    """load a well formed `npy` file representing a structured array

    Returns
    -------
        The array, the description, field names and their size.
    """
    a = np.load(f_name)
    if all_info:
        desc = a.dtype.descr
        nms = a.dtype.names
        sze = [i[1] for i in a.dtype.descr]
        return a, desc, nms, sze
    else:
        return a


# ----------------------------------------------------------------------
# (2) read_npy .... code section ---
def save_npy(a, f_name):
    """Save an array as an npy file.

    The type of data in each column is arbitrary.  It will be cast to the
    given dtype at runtime
    """
    np.save(f_name, a)


# ----------------------------------------------------------------------
# (3) read_txt .... code section ---
def read_txt(name="arr.txt"):
    """Read the structured/recarray created by save_txt.

    dtype : data type
        If `None`, it allows the structure to be read from the array.

    delimiter : string
        Use a comma delimiter by default.

    names : boolean
        If `True`, the first row contains the field names.

    see np.genfromtxt for all *args and **kwargs.
    """
    a = np.genfromtxt(name, dtype=None, delimiter=",",
                      names=True, autostrip=True)  # ,skip_header=1)
    return a


# ----------------------------------------------------------------------
# (4) save_txt .... code section ---
def save_txt(a, name="arr.txt", sep=", ", dt_hdr=True):
    """Save a NumPy structured, recarray to text.

    Requires:
    --------
    a     : array
        input array
    fname : filename
        output filename and path otherwise save to script folder
    sep   : separator
        column separater, include a space if needed
    dt_hdr: boolean
        if True, add dtype names to the header of the file
    """
    a_names = ", ".join(i for i in a.dtype.names)
    hdr = ["", a_names][dt_hdr]  # use "" or names from input array
    s = np.array(a.tolist(), dtype=np.string_)
    widths = [max([len(i) for i in s[:, j]])
              for j in range(s.shape[1])]
    frmt = sep.join(["%{}s".format(i) for i in widths])
    # vals = ", ".join([i[1] for i in a.dtype.descr])
    np.savetxt(name, a, fmt=frmt, header=hdr, comments="")
    print("\nFile saved...")


# ----------------------------------------------------------------------
# (5) arr_json .... code section ---
def arr_json(file_out, arr=None):
    """Send an array out to json format. Use json_arr to read the file.
    No error checking
    """
    import json
    import codecs
    json.dump(arr.tolist(), codecs.open(file_out, 'w', encoding='utf-8'),
              sort_keys=True, indent=4)
    # ----


# ----------------------------------------------------------------------
# (6) batch load and save to/from arrays and rasters
def array2raster(a, folder, fname, LL_corner, cellsize):
    """It is easier if you have a raster to derive the values from.

    >>> # Get one of the original rasters since they will have the same
    >>> # extent and cell size needed to produce output
    >>> r01 = rasters[1]
    >>> rast = arcpy.Raster(r01)
    >>> lower_left = rast.extent.lowerLeft
    >>> # this is a Point object... ie LL = arcpy.Point(10, 10)
    >>> cell_size = rast.meanCellHeight  # --- we will use this for x and y
    >>> f = r'c:/temp/result.tif'  # --- don't forget the extention

    Requires:
    ---------

    `arcpy` and `os` if not previously imported
    """
    if 'arcpy' not in list(locals().keys()):
        import arcpy
    if not os.path.exists(folder):
        return None
    r = arcpy.NumPyArrayToRaster(a, LL_corner, cellsize, cellsize)
    f = os.path.join(folder, fname)
    r.save(f)
    print("Array saved to...{}".format(f))


# ----------------------------------------------------------------------
# (7) batch load and save to/from arrays and rasters
def rasters2nparray(folder=None, to3D=False):
    """Batch the RasterToNumPyArray arcpy function to produce 3D or a list
    of 2D arrays

    NOTE:
    ----
        Edit the code... far simpler than accounting for everything.
        There is a reasonable expectation that rasters exist in the folder.

    Requires:
    --------
    modules :
        os, arcpy if not already loaded
    folder : folder
        A folder on disk... a real one
    to3D : boolean
        If False, a list of arrays, if True a 3D array
    """
    if 'arcpy' not in list(locals().keys()):
        import arcpy
    arrs = []
    if folder is None:
        return None
    if not os.path.exists(folder):
        return None
    arcpy.env.workspace = folder
    rasters = arcpy.ListRasters("*", "TIF")
    for raster in rasters:
        arrs.append(arcpy.RasterToNumPyArray(raster))
    if to3D:
        return np.array(arrs)
    else:
        return arrs


def _demo_a_io():
    """
    : -
    """
    _npy_file = "/Data/sample_20.npy"  # change to one in the Data folder
    _npy_file = "{}".format(script.replace("a_io.py", _npy_file))
    return _npy_file


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    fname = _demo_a_io()
