import trimesh
import numpy as np
import time
import paths
 # https://towardsdatascience.com/how-to-voxelize-meshes-and-point-clouds-in-python-ca94d403f81d


def scale_and_vox(mesh_path):
    mesh = trimesh.load(mesh_path)

    bounds = mesh.bounds
    size = bounds[1, :] - bounds[0, :]

    nominal_mesh_size = 1.0
    nominal_voxel_size = 0.002
    min_scale = max(size / nominal_mesh_size)
    desired_voxel_size = min_scale * nominal_voxel_size

    start = time.time()
    angel_voxel = mesh.voxelized(pitch=desired_voxel_size, method="ray") # ray, subdivide, binvox
    print(time.time() - start)
    print("---")

    # generate a voxelized mesh from the voxel grid representation, using the calculated colors
    voxelized_mesh = angel_voxel.as_boxes(colors=np.array([200, 50, 50, 150]))

    # Initialize a scene
    s = trimesh.Scene()
    s.add_geometry(mesh)
    # Add the voxelized mesh to the scene. If want to also show the intial mesh uncomment the second line and change the alpha channel of in the loop to something <100
    s.add_geometry(voxelized_mesh)
    # s.add_geometry(mesh)
    s.show()

    return angel_voxel



if __name__=="__main__":
    # mesh_path = 'stls/crane.stl'
    # mesh_path = paths.STL_PATH + "solid_1.stl"
    # angel_voxel = scale_and_vox(mesh_path)

    for i in range(20):
        mesh_path = paths.ONSHAPE_STL_PATH + "solid_" + str(i) + ".stl"
        angel_voxel = scale_and_vox(mesh_path)

