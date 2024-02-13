# if you have pyembree this should be decently fast
import matplotlib.pyplot as plt
import trimesh
import trimesh_util
import numpy as np
import paths
import random
import stopwatch
from tqdm import tqdm

# no overhang is 0, full overhang is pi/2
def get_overhangs(trimesh_aux: trimesh_util.MeshAuxilliaryInfo, cutoff_angle_rad=np.pi/2.0):
    samples, normals = trimesh_aux.sample_and_get_normals()
    sample_z = normals[:, 2]
    sample_angles = np.arcsin(sample_z) # overhang angles will be < 0
    samples_above_floor = samples[:, 2] > (0.4 + trimesh_aux.bound_lower[2])
    overhang_indices = np.logical_and(sample_angles > -np.pi/2.0, sample_angles < -cutoff_angle_rad)
    overhang_indices = np.logical_and(overhang_indices, samples_above_floor)

    overhang_samples = samples[overhang_indices]
    overhang_angles = -sample_angles[overhang_indices]

    return overhang_samples, overhang_angles

def get_stairsteps(trimesh_aux: trimesh_util.MeshAuxilliaryInfo, cutoff_angle_rad=np.pi/2.0):
    samples, normals = trimesh_aux.sample_and_get_normals()
    sample_z = normals[:, 2]
    sample_angles = np.arcsin(sample_z) # overhang angles will be < 0
    samples_above_floor = samples[:, 2] > (0.4 + trimesh_aux.bound_lower[2])
    overhang_indices = np.logical_and(sample_angles < np.pi/2.0 * 0.99, sample_angles > cutoff_angle_rad)
    overhang_indices = np.logical_and(overhang_indices, samples_above_floor)

    overhang_samples = samples[overhang_indices]
    overhang_angles = sample_angles[overhang_indices]

    return overhang_samples, overhang_angles

if __name__ == "__main__":
    ## Single STL
    # mesh_path = paths.get_thingiverse_stl_path(7345)
    mesh_path = paths.HOME_PATH + 'stls/crane.stl'
    mesh = trimesh.load(mesh_path)
    # mesh = trimesh_util.TRIMESH_TEST_MESH

    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    points, values = get_stairsteps(mesh_aux, np.pi/4)
    trimesh_util.show_sampled_values(mesh, points=points, values=values)

    ## Multi STL
    # for i in range(20):
    #     mesh_path = paths.get_onshape_stl_path(random.randint(1, 300))
    #     mesh = trimesh.load(mesh_path)
    #     calculate_and_show_gap(mesh)
