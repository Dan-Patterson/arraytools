# -*- coding: utf-8 -*-
"""
======
fc_geo
======

Script :  fc_geo.py

Author :   Dan_Patterson@carleton.ca

Modified : 2019-04-24

Purpose :  Poly features as arrays and their properties

Notes
-----
This script can be used to call the np_geo.py script which contains the Geo
class. This class is used to access the geometry properties of featureclasses.

References
----------
`Keeping order in sorted unique values
<https://stackoverflow.com/questions/15637336/numpy-unique-with-order-
preserved>`_.  useful for keeping order of sorted unique values

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=E0611  # stifle the arcgisscripting
# pylint: disable=E1101  # ditto for arcpy
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect
# pylint: disable=W0621  # redefining name

import sys
from textwrap import dedent
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as stu

import arcpy
from npGeo import Geo

#from arraytools.geom_common import _view_

script = sys.argv[0]

null_pnt = (np.nan, np.nan)  # ---- a null point

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=120, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -


# ===========================================================================
# ---- featureclass section, arcpy dependent via arcgisscripting
#
def _make_nulls_(in_fc, int_null=-999):
    """Return null values for a list of fields objects, excluding objectid
    and geometry related fields.  Throw in whatever else you want.

    Parameters
    ----------
    in_flds : list of arcpy field objects
        Use arcpy.ListFields to get a list of featureclass fields.
    int_null : integer
        A default to use for integer nulls since there is no ``nan`` equivalent
        Other options include

    >>> np.iinfo(np.int32).min # -2147483648
    >>> np.iinfo(np.int16).min # -32768
    >>> np.iinfo(np.int8).min  # -128

    >>> [i for i in cur.__iter__()]
    >>> [[j if j else -999 for j in i] for i in cur.__iter__() ]
    """
    nulls = {'Double': np.nan, 'Single': np.nan, 'Float': np.nan,
             'Short': int_null, 'SmallInteger': int_null, 'Long': int_null,
             'Integer': int_null, 'String':str(None), 'Text':str(None),
             'Date': np.datetime64('NaT')}
    #
    desc = arcpy.da.Describe(in_fc)
    if desc['dataType'] != 'FeatureClass':
        print("Only Featureclasses are supported")
        return None, None
    in_flds = desc['fields']
    shp = desc['shapeFieldName']
    good = [f for f in in_flds if f.editable and f.name != shp]
    fld_dict = {f.name: f.type for f in good}
    fld_names = list(fld_dict.keys())
    null_dict = {f: nulls[fld_dict[f]] for f in fld_names}
    # ---- insert the OBJECTID field
    return null_dict, fld_names

   
def fc_composition(in_fc, SR=None):
    """Featureclass geometry composition in terms of shapes, shape parts, and
    point counts for each part
    """
    if SR is None:
        SR = getSR(in_fc)    
    with arcpy.da.SearchCursor(in_fc, 'SHAPE@', spatial_reference=SR) as cur:
        len_lst = []
        for i, row in enumerate(cur):
            p = row[0]
            parts = p.partCount
            num_pnts = np.asarray([p[i].count for i in range(parts)])
            IDs = np.repeat(i, parts)
            part_count = np.arange(parts)
            result = np.stack((IDs, part_count, num_pnts), axis=-1)
            len_lst.append(result)
    return len_lst


def getSR(in_fc, verbose=False):
    """Return the spatial reference of a featureclass"""
    desc = arcpy.da.Describe(in_fc)
    SR = desc['spatialReference']
    if verbose:
        print("SR name: {}  factory code: {}".format(SR.name, SR.factoryCode))
    return SR

# ==========================
# Option 1 ----
#
def fc_shapes(in_fc, SR=None):
    """Featureclass to arcpy shapes.  Returns polygon, polyline, multipoint,
    or points.
    """
    if SR is None:
        SR = getSR(in_fc)    
    with arcpy.da.SearchCursor(in_fc, 'SHAPE@', spatial_reference=SR) as cur:
        out = [row[0] for row in cur]
    return out

# Option 2 ----
#
def fc2geo_inter(in_fc, SR=None):
    """Derive, geometry objects from a featureClass searchcursor in json-like
    format. ** really slow **

    Parameters
    ----------
    in_fc : text
        Path to the input featureclass

    Returns
    -------
    arcpy geometry objects

    >>> g0, g1, g2 = polys = cur2geo(in_fc)  # ---- 3 polygons
    >>> g0
        {'type': 'MultiPolygon',
         'coordinates': [[[(300020.0, 5000000.0), (300010.0, 5000000.0),
                            ... snip ..., (300020.0, 5000000.0)]]]}
    """
    if SR is None:
        SR = getSR(in_fc)
    with arcpy.da.SearchCursor(in_fc, 'SHAPE@', None, SR) as cursor:
        a = [np.asarray(row[0].__geo_interface__['coordinates'])
             for row in cursor]
    return a

# Option 3 ----
#
def fc_arc_array(in_fc, SR=None):
    """fc to arcpy Array"""
    if SR is None:
        SR = getSR(in_fc)
    z = arcpy.da.FeatureClassToNumPyArray(in_fc,
                                          ['OID@', 'SHAPE@X', 'SHAPE@Y'],
                                          "", SR,
                                          explode_to_points=True)
    idz = z['OID@']
    xy = stu(z[['SHAPE@X', 'SHAPE@Y']])
    idbin = np.cumsum(np.bincount(idz))
    m = np.nanmin(xy, axis=0)
    a0 = xy - m
    return idz, idbin, xy, a0

# Option 4 ----
#
def fc_as_narray(in_fc, SR=None):
    """largely for testing"""
    if SR is None:
        SR = getSR(in_fc)
    with arcpy.da.SearchCursor(in_fc, ['OID@', 'SHAPE@XY'],
                               "", SR,
                               explode_to_points=True) as cursor:
        arr = cursor._as_narray()
    return arr

# Option 5 ----  The one I am using
#
def fc_geometry(in_fc, SR=None):
    """Derive, arcpy geometry objects from a featureClass searchcursor.

    Parameters
    ----------
    in_fc : text
        Path to the input featureclass.  Points not supported.
    SR : spatial reference
       Spatial reference object, name or id

    Returns
    -------
    ``a_2d, IFT`` (ids_from_to), where a_2d are the points as a 2D array,
    ``IFT``represent the id numbers (which are repeated for multipart shapes),
    and the from-to pairs of the feature parts.

    See Also
    --------
    Use ``arrays_Geo`` to produce ``Geo`` objects directly pre-existing arrays,
    or arrays derived form existing arcpy poly objects which originated from
    esri featureclasses.

    Notes
    -----
    Multipoint, polylines and polygons and its variants are supported.

    **Point and Multipoint featureclasses**

    >>> cent = arcpy.da.FeatureClassToNumPyArray(pnt_fc,
                                             ['OID@', 'SHAPE@X', 'SHAPE@Y'])

    For multipoints, use

    >>> allpnts = arcpy.da.FeatureClassToNumPyArray(multipnt_fc,
                                                ['OID@', 'SHAPE@X', 'SHAPE@Y']
                                                explode_to_points=True)

    **IFT array structure**

    To see the ``IFT`` output as a structured array, use the following.

    >>> dt = np.dtype({'names': ['ID', 'From', 'To'], 'formats': ['<i4']*3})
    >>> z = IFT.view(dtype=dt).squeeze()
    >>> prn_tbl(z)  To see the output in tabular form

    **Flatten geometry tests**

    >>> %timeit fc_geometry(in_fc2, SR)
    105 ms ± 1.04 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    ...
    >>> %%timeit
    ... cur = arcpy.da.SearchCursor(in_fc2, 'SHAPE@', None, SR)
    ... p = [row[0] for row in cur]
    ... sh = [[i for i in itertools.chain.from_iterable(shp)] for shp in p]
    ... pnts = [[[pt.X, pt.Y] if pt else null_pnt for pt in lst] for lst in sh]
    4.4 ms ± 21.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    msg = """
    Use arcpy.FeatureClassToNumPyArray for Point files.
    MultiPoint, Polyline and Polygons and its variants are supported.
    """
    # ----
    def _multipnt_(in_fc, SR):
        """Convert multipoint geometry to array"""
        pnts = arcpy.da.FeatureClassToNumPyArray(
                   in_fc, ['OID@', 'SHAPE@X', 'SHAPE@Y'],
                   spatial_reference=SR,
                   explode_to_points=True)
        id_len = np.vstack(np.unique(pnts['OID@'], return_counts=True)).T
        a_2d = stu(pnts[['SHAPE@X', 'SHAPE@Y']])  # ---- use ``stu`` to convert
        return id_len, a_2d
    # ----
    def _polytypes_(in_fc, SR):
        """Convert polylines/polygons geomeetry to array"""
        import json
        def _densify_curves_(geom, deg=1):
            """densify geometry for circle and ellipse (geom) at ``deg`` degree
            increments. deg, angle = (1, 361), (2, 181), (5, 73)
            """
            j = json.loads(geom.JSON)
            has_curves = np.any(['curve' in i for i in list(j.keys())])
            if has_curves:
                return geom.densify('ANGLE', 1, np.deg2rad(deg))
            return geom
        # ----
        null_pnt = (np.nan, np.nan)
        id_len = []
        a_2d = []
        with arcpy.da.SearchCursor(in_fc, ('OID@', 'SHAPE@'), None, SR) as cursor:
            for p_id, row in enumerate(cursor):
                sub = []
                IDs = []
                num_pnts = []
                p_id = row[0]
                geom = row[1]
                prt_cnt = geom.partCount
                p_num = geom.pointCount  # ---- added
                if (prt_cnt == 1) and (p_num <= 4):
                    geom = _densify_curves_(geom)
                for arr in geom:
                    pnts = [[pt.X, pt.Y] if pt else null_pnt for pt in arr]
                    sub.append(np.asarray(pnts))
                    IDs.append(p_id)
                    num_pnts.append(len(pnts))
                part_count = np.arange(prt_cnt)
                #too = np.cumsum(num_pnts)
                result = np.stack((IDs, part_count, num_pnts), axis=-1)
                id_len.append(result)
                a_2d.extend([j for i in sub for j in i])
        # ----
        id_len = np.vstack(id_len)  #np.array(id_len)
        a_2d = np.asarray(a_2d)
        return id_len, a_2d

def fc_data(in_fc):
    """Pull all editable attributes from a featureclass tables.  During the
    process, <null> values are changed to an appropriate type.

    Parameters
    ----------
    in_fc : text
        Path to the input featureclass

    Notes
    -----
    The output objectid and geometry fields are renamed to
    `OID_`, `X_cent`, `Y_cent`, where the latter two are the centroid values.
    """
    flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']
    null_dict, fld_names = _make_nulls_(in_fc, int_null=-999)
    fld_names = flds + fld_names
    new_names = ['OID_', 'X_cent', 'Y_cent']
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, fld_names,
                                          skip_nulls=False,
                                          null_value=null_dict)
    a.dtype.names = new_names + fld_names[3:]
    return np.asarray(a)


# ===========================================================================
# ---- back to featureclass 
#
def _arr_poly_(arr, SR, as_type):
    """Single array to polygon.  The array can be multipart with or without
    interior rings.  Outer rings are ordered clockwise, inner rings (holes)
    are ordered counterclockwise.  For polylines, there is no concept of order.
    Splitting is modelled after _nan_split_(arr)

    Parameters
    ----------
    arr : array
        Points array
    SR : spatial reference
       Spatial reference object, name or id
    as_type : text
        Polygon or Polyline
    """
    subs = []
    s = np.isnan(arr[:, 0])
    if np.any(s):
        w = np.where(s)[0]
        ss = np.split(arr, w)
        subs = [ss[0]]
        subs.extend(i[1:] for i in ss[1:])
    else:
        subs.append(arr)
    aa = []
    for sub in subs:
        aa.append([arcpy.Point(*pairs) for pairs in sub])
    if as_type == 'POLYGON':
        poly = arcpy.Polygon(arcpy.Array(aa), SR)
    elif as_type == 'POLYLINE':
        poly = arcpy.Polyline(arcpy.Array(aa), SR)
    return poly


def array_poly(a, p_type, sr, ids, from_to):
    """assemble the polygons separately from geometry_fc"""
    def _poly_pieces_(a, from_to):
        """deconstruct the 2D array into its pieces
        """
        return [a[f:t] for f, t in from_to]
    # ----
    if ids is None:
        ids = np.arange(len(a)).tolist()
    chunks = _poly_pieces_(a, from_to)  # ---- _poly_pieces_ chunks input
    polys = []
    for i in chunks:
        p = _arr_poly_(i, sr, p_type)  # ---- _arr_poly_ makes parts of chunks
        polys.append(p)
    out = list(zip(polys, ids))
    return out


def geometry_fc(a, ids, from_to, p='POLYGON', gdb=None, fname=None, sr=None):
    """Reform poly features from the list of arrays created by ``fc_geometry``.

    Parameters
    ----------
    a : array or list of arrays
        Some can be object arrays, normally created by ``pnts_arr``
    ids : list/array
        Identifies which feature each input belongs to.  This enables one to
        account for multipart shapes
    from_to : list/array
        See ids above, denotes the actual splice elements for each feature.
    p : string
        Uppercase geometry type
    gdb : text
        Geodatabase name
    fname : text
        Featureclass name
    sr : spatial reference
        name or object
.
    ``arr_poly`` is required

    Returns
    -------
    Singlepart and multipart featureclasses
    """
    out = array_poly(a, p, sr, ids, from_to)   # call array_poly and ist sub
    name = gdb + "\\" + fname
    wkspace = arcpy.env.workspace = 'in_memory'
    arcpy.management.CreateFeatureclass(wkspace, fname, p,
                                        spatial_reference=sr)
    arcpy.management.AddField(fname, 'ID_arr', 'LONG')
    with arcpy.da.InsertCursor(fname, ['SHAPE@', 'ID_arr']) as cur:
        for row in out:
            cur.insertRow(row)
    out_fname = fname + "_mp"
    arcpy.management.Dissolve(fname, out_fname, "ID_arr",
                              multi_part="MULTI_PART",
                              unsplit_lines="DISSOLVE_LINES")
    arcpy.management.CopyFeatures(out_fname, name)

#
# ============================================================================
# ---- array dependent
def prn_q(a, edges=3, max_lines=25, width=120, decimals=2):
    """Format a structured array by setting the width so it hopefully wraps.
    """
    width = min(len(str(a[0])), width)
    with np.printoptions(edgeitems=edges, threshold=max_lines, linewidth=width,
                         precision=decimals, suppress=True, nanstr='-n-'):
        print("\nArray fields/values...:")
        print("  ".join([n for n in a.dtype.names]))
        print(a)


def _nan_split_(arr):
    """Split at an array with nan values for an  ndarray.  It is assumed that
    the `x` column contains nan to separate array parts.

    >>> z.T
    array([[20., 20., 10., 10., nan, 20., 20., 10., 10.],  # Xs
           [10.,  0.,  0., 10., nan, 10.,  0.,  0., 10.]]) # Ys
    >>> _nan_split_(z)  # shown side-by-side
    array([[[20., 10.],   [[20., 10.],
            [20.,  0.],    [20.,  0.],
            [10.,  0.],    [10.,  0.],
            [10., 10.]],   [10., 10.]]])
    """
    s = np.isnan(arr[:, 0])                 # nan is used to split the 2D arr
    if np.any(s):
        w = np.where(s)[0]
        ss = np.split(arr, w)
        subs = [ss[0]]                      # collect the first full section
        subs.extend(i[1:] for i in ss[1:])  # slice off nan from the remaining
        return np.asarray(subs)
    return arr


def pieces(arr):
    """split up objects arrays into a list of pieces

    casting {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}
    """
    out = []
    for item in arr:
        check = np.any(np.isnan(item))
        if check:
            subs = _nan_split_(item)
            for sub in subs:
                out.append(sub)
        else:
            out.append(item)
    return out


# ==== poly featureclass to arrays ==========================================
#      arrays to poly featureclass
#
def p2p_(poly):
    """See poly_pnts for details.  This is for single poly feature conversion
    requires null_pnt = (np.nan, np.nan)
    """
    sub = []
#    for i, arr in enumerate(poly):
    for arr in poly:
        pnts = [[pt.X, pt.Y] if pt else null_pnt for pt in arr]
        sub.append(np.asarray(pnts)) #append(pnts)
    return np.asarray(sub)


# ============================================================================
# ---- Extras..... property calculations assumed in Geo class in np_geo.py
#
def _poly_pieces_(a, from_to):
    """deconstruct the 2D array into its pieces
    """
    return [a[f:t] for f, t in from_to]

def _pp_(poly):
    """poly to points worker """
    sub = []
    for arr in poly:
        pnts = [[pt.X, pt.Y] if pt else null_pnt for pt in arr]
        sub.append(np.asarray(pnts))
    return np.asarray(sub)

def _e_area(a):
    """mini e_area with a twist, shoelace formula using einsum.
    Used by poly_area and poly_centroid"""
    x0, y1 = (a.T)[:, 1:]
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)
    e1 = np.einsum('...i,...i->...i', x1, y1)
    return np.nansum((e0-e1)*0.5)


def poly_areas(a, ids, from_to):
    """Calculate the area of a 2D array representing a polygon.

    If the polygon is multipart and/or has inner rings, it is sliced into
    sections using indices of the bounds of each part as represented by
    ``from_to``.

    Array ``a`` is created from ``poly_pnts`` and all the required components
    are generated by that function.
    """
    # ----
    boot = any([not isinstance(var, (list, tuple, np.ndarray))
                for var in (a, ids, from_to)])
    if boot:
        return None
    subs = [_e_area(a[f:t]) for f, t in from_to]  # _poly_pieces_ equivalent
    totals = np.bincount(ids, weights=subs)
    return totals


def poly_centroids(a, ids, from_to):
    """Return the centroid of a closed polygon.

    Parameters
    ----------
    a : array
        A 2D array of point coordinates.  You need to keep the duplicate
        first and last point.
    a_6 : number
        If area has been precalculated, you can use its value.
    e_area : function (required)
        Contained in this module.
    """
    def _cal_(p):
        """ """
        x, y = p.T
        t = ((x[:-1] * y[1:]) - (y[:-1] * x[1:]))
        a_6 = _e_area(p) * 6.0  # area * 6.0
        x_c = np.nansum((x[:-1] + x[1:]) * t, axis=0) / a_6
        y_c = np.nansum((y[:-1] + y[1:]) * t, axis=0) / a_6
        return np.asarray([-x_c, -y_c]), a_6
    #
    def weighted(xy, ids, areas):
        """ weighted coordinate by area, xy is either the x or y coordinate
        """
        w = xy * areas             # area weighted x or y
        w1 = np.bincount(ids, w)   # weight divided by bin size
        ar = np.bincount(ids, areas) # areas per bin
        return w1/ar
    #
    pieces = _poly_pieces_(a, ids, from_to)
    centroids = []
    areas = []
    for p in pieces:
        cen, area = _cal_(p)
        centroids.append(cen)
        areas.append(area)
    centroids = np.asarray(centroids)
    areas = np.asarray(areas)
    xs = weighted(centroids[:, 0], ids, areas)
    ys = weighted(centroids[:, 1], ids, areas)
    return np.array(list(zip(xs, ys)))


def poly_extents(a, ids, from_to):
    """Extents are returns as L(eft), B(ottom), R(ight), T(op)
    """
    def _extent_(i):
        """extent of a sub-array in an object array"""
        return np.concatenate((np.nanmin(i, axis=0),
                               np.nanmax(i, axis=0)))
    # ----
    p_ext = []
    uni = np.unique(ids)
    ft0 = [np.ravel(from_to[np.where(ids == i)]) for i in uni]
    ft1 = np.array([[i.min(), i.max()] for i in ft0])
    for f, t in ft1:  #from_to:
        LBRT = _extent_(a[f:t])
        p_ext.append(LBRT)
    p_ext = np.asarray(p_ext)
    return p_ext


def poly_lengths(a, ids, from_to):
    """Calculate total and segment lengths for poly features.

    Parameters
    ----------
    a : array
        A 2D array representing poly feature coordinates.
    ids : list/array
        The array may contain more than one feature, and each feature can be
        segmented
    from_to : list/array
        Features are separated according to the ``from_to`` sequences.

    The ids represent the id numbers of each poly and its part.  Duplication of
    a number means that it has more than one part (eg. polys 1, 2 have 2 parts)

    >>> ids # ==> array([0, 1, 1, 2, 2, 3, 4])
    >>> from_to.T  # array([[ 0,  5, 16, 26, 36, 48, 57],   # from
    ...                   [ 5, 16, 26, 36, 48, 57, 65]])  # to
    """
    def _cal(diff):
        """ perform the calculation, see above
        """
        d_leng = np.sqrt(np.einsum('ij,ij->i', diff, diff)).squeeze()
        length = np.nansum(d_leng.flatten())
        return length, d_leng
    # ----
    tot_leng = []
    seg_leng = []
    for f, t in from_to:
        sub = a[f:t]
        diff = sub[:-1] - sub[1:]
        length, d_leng = _cal(diff)
        tot_leng.append(length)
        seg_leng.append(d_leng)
    totals = np.bincount(ids, weights=tot_leng)
    return totals, seg_leng
# ===========================================================================
# ---- main section: testing or tool run ------------------------------------

def _demo_():
    """
    Test values
    gdb = r"C:/Arc_projects/CoordGeom/CoordGeom.gdb"
    p0, p1, p2, p3, p4 = polys
    """
#    in_fc = r"C:/Arc_projects/CoordGeom/CoordGeom.gdb/Polygons"
    in_fc = r"C:\Arc_projects\Canada\Canada.gdb\Ontario_LCConic"
    SR = getSR(in_fc)
    polys = fc_shapes(in_fc)
    # ---- Do the work ----
    tmp, IFT = fc_geometry(in_fc)
    m = np.nanmin(tmp, axis=0)
    a = tmp  - m
    #a1 = np.asarray([a[f:t] for f, t in from_to])
    #p_arr = [_arr_poly_(i, SR) for i in a1]  # ** to reverse np.concatenate(a1)
    #p_arr = None
    frmt = """
    Polygon
    ids-from-to:
    {}
    """
    print(dedent(frmt).format(IFT))
    #arr_poly_fc(a1, p_type='POLYGON', gdb=gdb, fname='a1_test', sr=SR, ids=ids)
    return in_fc, SR, polys, tmp, IFT, a


testing = False
if testing:
    in_fc, SR, polys, tmp, IFT, a = _demo_()
    #from npGeo import Geo
    z = Geo(a, IFT)
else:
    print("Running from tool")


# ==== Processing finished ====
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
