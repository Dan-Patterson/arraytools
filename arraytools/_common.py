# -*- coding: UTF-8 -*-
"""
:Script:   _common.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-11-03
:Purpose:  common tools for working with numpy arrays and featureclasses
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all_common__ = ['_describe', 'fc_info', 'fld_info', 'tweet']

# ----------------------------------------------------------------------------
# ---- Geometry objects and generic geometry/featureclass functions ----------
# ----------------------------------------------------------------------------
def _describe(in_fc):
    """Simply return the arcpy.da.Describe object
    : desc.keys() an abbreviated list...
    : [... 'OIDFieldName'... 'areaFieldName', 'baseName'... 'catalogPath',
    :  ... 'dataType'... 'extent', 'featureType', 'fields', 'file'... 'hasM',
    :  'hasOID', 'hasZ', 'indexes'... 'lengthFieldName'... 'name', 'path',
    :  'rasterFieldName', ..., 'shapeFieldName', 'shapeType',
    :  'spatialReference',  ...]
    """
    return arcpy.da.Describe(in_fc)


def fc_info(in_fc, prn=False):
    """Return basic featureclass information, including...
    :
    : shp_fld  - field name which contains the geometry object
    : oid_fld  - the object index/id field name
    : SR       - spatial reference object (use SR.name to get the name)
    : shp_type - shape type (Point, Polyline, Polygon, Multipoint, Multipatch)
    : - others: 'areaFieldName', 'baseName', 'catalogPath','featureType',
    :   'fields', 'hasOID', 'hasM', 'hasZ', 'path'
    : - all_flds =[i.name for i in desc['fields']]
    """
    desc = _describe(in_fc)
    args = ['shapeFieldName', 'OIDFieldName', 'shapeType', 'spatialReference']
    shp_fld, oid_fld, shp_type, SR = [desc[i] for i in args]
    if prn:
        frmt = "FeatureClass:\n   {}".format(in_fc)
        f = "\n{!s:<16}{!s:<14}{!s:<10}{!s:<10}"
        frmt += f.format(*args)
        frmt += f.format(shp_fld, oid_fld, shp_type, SR.name)
        tweet(frmt)
    else:
        return shp_fld, oid_fld, shp_type, SR


def fld_info(in_fc, prn=False):
    """Field information for a featureclass.
    :  prn=True - returns the values
    :  prn=False - simply prints the results
    :field properties...
    :  'aliasName', 'baseName', 'defaultValue', 'domain', 'editable',
    :  'isNullable', 'length', 'name', 'precision', 'required', 'scale', 'type'
    """
    flds = arcpy.ListFields(in_fc)
    f_info = [(i.name, i.type, i.length) for i in flds]
    if prn:
        frmt = "FeatureClass:\n   {}\n".format(in_fc)
        frmt += "{!s:<14}{!s:<10}{!s:<6}\n".format("Name", "Type", "Length")
        f = "{!s:<14}{!s:<10}{!s:>6}"
        frmt += "\n".join([f.format(*i) for i in f_info])
        tweet(frmt)
    else:
        return f_info


def tweet(msg):
    """Print a message for both arcpy and python.
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)
    print(arcpy.GetMessages())


def _demo():
    """
    : -
    """
    pass


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    _demo()
