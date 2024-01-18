import open3d as o3d
import numpy as np

from pc_skeletor import Dataset

# downloader = Dataset()
# trunk_pcd_path, branch_pcd_path = downloader.download_semantic_tree_dataset()

print("Reading mesh")
mesh = o3d.io.read_triangle_mesh("stls/Antenna_DJI.stl")
num_elements = len(mesh.triangles)
print(num_elements)
print("Converting to point cloud")
pcd = mesh.sample_points_poisson_disk(50000)

# pcd_trunk = o3d.io.read_point_cloud(trunk_pcd_path)
# pcd_branch = o3d.io.read_point_cloud(branch_pcd_path)
# pcd = pcd_trunk + pcd_branch

from pc_skeletor import LBC

print("Extracting")
lbc = LBC(point_cloud=pcd,
          down_sample=0.008)
lbc.extract_skeleton()
lbc.extract_topology()

print("visualize")
lbc.visualize()
# lbc.show_graph(lbc.skeleton_graph)
# lbc.show_graph(lbc.topology_graph)
# lbc.save('./output')
# lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
#             steps=300,
#             output='./output')