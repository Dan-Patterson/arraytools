**Data files**

Sample npy files to use with functions

sample_20.npy, sample_1000.npy and sample_10K.npy basically containe random data with the following dtype.

```
a[:3]
 
array([(1, 0, 'B', 'B_', 'Hall', 11),
       (2, 1, 'A', 'A_', 'Hall', 24),
       (3, 2, 'C', 'C_', 'Hosp', 43)],
      dtype=[('OBJECTID', '<i4'), ('f0', '<i4'), ('County', '<U2'),
             ('Town', '<U6'), ('Facility', '<U8'), ('Time', '<i4')])
```
Useful for testing purposes.

xyz.npy  58, 3D points on a transect line
```
a[:3]
 
array([(1,  7.47, 1016.11, 1.98),
       (2, 19.02, 1008.64, 3.96),
       (3, 31.92,  999.13, 6.14)],
      dtype=[('OBJECTID', '<i4'), ('POINT_X', '<f8'), ('POINT_Y', '<f8'), ('POINT_Z', '<f8')])
```
