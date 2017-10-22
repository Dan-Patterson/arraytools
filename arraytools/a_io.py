# -*- coding: UTF-8 -*-
"""
:Script:   a_io.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-10-17
:Purpose: basic io tools for numpy arrays and operating system functions
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import os
import numpy as np


ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['load_npy',  # load numpy npy files
           'save_npy',  # save array to npy format
           'read_txt',  # read a text formatted array
           'save_txt',  # save array to text format
           '_arr_json',  # array to json file
           '_get_dir',  # various function for accessing folders
           'all_folders',
           'sub_folders']


# ----------------------------------------------------------------------
# (1) load_npy .... code section ---
def load_npy(f_name, all_info=False):
    """load a well formed npy file representing a structured array
    :
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
# (1) read_npy .... code section ---
def save_npy(a, f_name):
    """Save an array as an npy file
    :    The type of data in each column is arbitrary
    :    It will be cast to the given dtype at runtime
    """
    np.save(f_name, a)


# ----------------------------------------------------------------------
# (2) read_txt .... code section ---
def read_txt(name="arr.txt"):
    """Read the structured/recarray created by save_txt.
    :  dtype=None ... allow the structure to be read from the array.
    :  delimiter  ... use a comma delimiter by default
    :  names=True ... first row contains the field names.
    :                 uncomment skip_header part if not needed.
    """
    a = np.genfromtxt(name, dtype=None, delimiter=",",
                      names=True, autostrip=True)  # ,skip_header=1)
    return a


# ----------------------------------------------------------------------
# (3) save_txt .... code section ---
def save_txt(a, name="arr.txt", sep=", ", dt_hdr=True):
    """Save a NumPy structured, recarray to text.
    :Requires:
    :--------
    :  a     : input array
    :  fname : output filename and path otherwise save to script folder
    :  sep   : column separater, include a space if needed
    :  dt_hdr: if True, add dtype names to the header of the file
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
# (4) save_txt .... code section ---
def _arr_json(file_out, arr=None):
    """Send an array out to json format. Use json_arr to read the file.
    :  no error checking
    """
    import json
    import codecs
    json.dump(arr.tolist(), codecs.open(file_out, 'w', encoding='utf-8'),
              sort_keys=True, indent=4)
    # ----


# ----------------------------------------------------------------------
# (5) general file functions ... code section ---
def _get_dir(path):
    """Get the directory list from a path, excluding geodatabase folders
    :  Used by.. print_folders
    """
    if os.path.isfile(path):
        path = os.path.dirname(path)
    p = os.path.normpath(path)
    full = [os.path.join(p, v) for v in os.listdir(p)]
    dirlist = [val for val in full if os.path.isdir(val)]
    return dirlist


def all_folders(path, first=True, prefix=""):
    """ Print recursive listing of contents of path
    :Requires: _get_dir
    :--------
    :Notes:
    :-----
    : useful.... cp = os.path.commonprefix(dirlist)
    """
    if first:  # Detect outermost call, print a heading
        print("Folder listing for....\n|--{}".format(path))
        prefix = "|-"
        first = False
        cprev = path
    dirlist = _get_dir(path)
    for d in dirlist:
        fullname = os.path.join(path, d)  # Turn name into full pathname
        if os.path.isdir(fullname):       # If a directory, recurse.
            cprev = path
            pad = ' ' * len(cprev)
            n = d.replace(cprev, pad)
            print(prefix + "-" + n)  # fullname) # os.path.relpath(fullname))
            p = "  "
            all_folders(fullname, first=False, prefix=p)
    # ----


def sub_folders(path):
    """print the folders in a path
    """
    import pathlib
    print("Path...\n{}".format(path))
    r = " "*len(path)
    f = "\n".join([(p._str).replace(path, r)
                   for p in pathlib.Path(path).iterdir() if p.is_dir()])
    print("{}".format(f))


def _demo():
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
    _npy_file = _demo()
#    x = "C:/Git_Dan/arraytools/Data/x.txt"
