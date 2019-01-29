**geomtools**

Reorganizing geometry functions

Includes
1. pip.py  
   - extent polygon  (extent_poly)
       
           create the bounding rectangle of a polygon's extent returning the left-bottom and top-right points

   - point in polygon (pnts_in_poly)
   
           runs the actual code for determining the points falling within a polygon. 

   - points in extent  (pnts_in_extent)
   
           prunes out points that don't fall within a polygon's extent as a prelude to \_crossing_num_
   
   - crossing number   (\_crossing_num_)
   
           for points that fall with the extent of a polygon, this does the final check
