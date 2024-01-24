# if you have pyembree this should be decently fast
import matplotlib.pyplot as plt
import trimesh
import trimesh_util
import numpy as np
import paths
import random
import stopwatch
from tqdm import tqdm

NO_GAP_VALUE = -1

def normalize_01(array: np.ndarray):
    copy = array.copy()
    copy -= np.amin(copy)
    copy /= np.amax(copy)
    return copy

def sample_and_get_normals(mesh, mesh_aux: trimesh_util.MeshAuxilliaryInfo):
    sample_points, face_index = trimesh.sample.sample_surface_even(mesh, 50000)
    normals = mesh_aux.facet_normals[face_index]
    return sample_points, normals

def show_sampled_thickness(mesh):
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    origins, normals = sample_and_get_normals(mesh, mesh_aux)

    facet_offset = -normals * 0.001
    hits, ray_ids, tri_ids = mesh.ray.intersects_location(ray_origins=origins + facet_offset,
                                        ray_directions=-normals,
                                        multiple_hits=False)

    hit_origins = origins[ray_ids]

    distances = np.linalg.norm(hits - hit_origins, axis=1)
    wall_thicknesses = distances

    show_sampled_values(mesh, points=hit_origins, values=wall_thicknesses)

def show_sampled_gaps(mesh):
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    origins, normals = sample_and_get_normals(mesh, mesh_aux)

    facet_offset = normals * 0.1 # This offset needs to be tuned based on stl dimensions
    hits, ray_ids, tri_ids = mesh.ray.intersects_location(ray_origins=origins + facet_offset,
                                        ray_directions=normals,
                                        multiple_hits=False)
    hit_origins = origins[ray_ids]
    distances = np.linalg.norm(hits - hit_origins, axis=1)
    gap_sizes = distances

    print("Num gap samples: ", len(gap_sizes[gap_sizes != NO_GAP_VALUE]))

    show_sampled_values(mesh, points=hit_origins, values=gap_sizes)

def show_sampled_values(mesh, points, values):
    s = trimesh.Scene()

    norm_values = normalize_01(values)

    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    colors = 255.0 * cmap(norm_values)
    colors[:, 3] = int(0.8 * 255)
    point_cloud = trimesh.points.PointCloud(vertices=points,
                                            colors=colors)
    s.add_geometry(point_cloud)
    s.add_geometry(mesh)
    s.show()

def sample_evenly_and_show(mesh):
    sample_points, face_index = trimesh.sample.sample_surface_even(mesh, 10000)

    samples_color = np.array([255, 0, 0, 255])
    point_cloud = trimesh.points.PointCloud(vertices=sample_points,
                                       colors=samples_color)

    s = trimesh.Scene()
    s.add_geometry(mesh)
    s.add_geometry(point_cloud)
    s.show()

if __name__ == "__main__":
    ## Single STL
    # mesh_path = paths.get_onshape_stl_path(185)
    # mesh_path = paths.get_thingiverse_stl_path(5561)
    # mesh_path = 'stls/crane.stl'
    # mesh = trimesh.load(mesh_path)
    # # mesh = trimesh_util.TRIMESH_TEST_MESH
    # show_sampled_gaps(mesh)
    # show_sampled_thickness(mesh)

    # ## Multi STL
    for i in range(20):
        random_index = random.randint(0, 10000)
        print(random_index)
        mesh_path = paths.get_thingiverse_stl_path(random_index)
        mesh = trimesh.load(mesh_path)
        # show_sampled_thickness(mesh)
        # sample_evenly_and_show(mesh)
        show_sampled_thickness(mesh)

