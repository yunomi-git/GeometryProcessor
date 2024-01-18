

import trimesh
import numpy as np
from shapely.geometry import LineString

mesh = trimesh.load('stls/Antenna_DJI.stl', force="mesh")
# mesh.show()

slice = mesh.section(plane_origin=mesh.centroid,
                     plane_normal=[1,0,1])

# slice.show()
slice_2D, to_3D = slice.to_planar()
all_slices = slice_2D
for i in range(10):
    slice = mesh.section(plane_origin=mesh.centroid,
                         plane_normal=[np.random.rand(), np.random.rand(), np.random.rand()])

    # slice.show()
    slice_2D, to_3D = slice.to_planar()
    all_slices = all_slices + slice_2D
    (slice_2D + slice_2D.medial_axis()).show()

# all_slices.show()

