Geometry Tools
==============

**geometry_common**

\_reshape_ and \_view_
```
a = np.array([(341000., 5021000.), (341000., 5022000.),
              (342000., 5022000.), (341000., 5021000.)],
              dtype=[('X', '<f8'), ('Y', '<f8')])

 _reshape_(a)
 
array([[ 341000., 5021000.],
       [ 341000., 5022000.],
       [ 342000., 5022000.],
       [ 341000., 5021000.]])
```

**geom_properties**

```
    'max_', 'median_', 'min_',      # max, mean, median, min
    'extent_',
    'center_', 'centers',           # centers, centroids
    'centroid_', 'centroids',
    'e_area', 'e_dist', 'e_leng',   # areas, distances, lengths
    'areas', 'lengths', 
    'total_length', 'seg_lengths',
    'dx_dy_np', 'angle_np',         # angles, direction
    'azim_np',  'angle_between',
    'angle_2pnts', 'angle_seq',
    'angles_poly',
    'orig_dest_angle', 'line_dir'       
```

**geom_create**

```
   'rot_matrix',                     # general
   'arc_', 'arc_sector',             # arcs, circle, ellipse
   'circle', 'circle_mini',
   'circle_ring', 'ellipse',
   'hex_flat', 'hex_pointy',         # hex
   'convex',                         # convex
   'mesh_xy',                        # mesh
   'pyramid',                        # pyramid, rectangle, triangle
   'rectangle',
   'triangle',
   'pnt_from_dist_bearing',          # points   
   'xy_grid',
   'transect_lines',                 # lines
   'spiral_archim'                   # esoterica
 ```
 
 **geom**
 
 ```
   'close_arr', 'stride',             # utilities
   'poly2segments', 
   'intersect_pnt', 'intersects',     # intersection
   'cartesian_dist',                  # distance, array combination
   'densify_by_distance',             # densify simplify
   'densify_by_factor', '\_convert',
   'densify', 'simplify',
   'rotate', 'trans_rot',             # translation, rotation
   'pnt_in_list',                     # spatial queries and analysis
   'pnt_on_seg', 'pnts_on_line',      # point construction
   'pnt_on_poly',
   'point_in_polygon',                # point, in, near etc
   'knn', 'nn_kdtree', 'cross',
   'remove_self',                     # not placed
   'adjacency_edge'
 ```
