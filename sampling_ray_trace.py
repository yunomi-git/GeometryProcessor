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

    # if len(hits) != len(origins):
    #     print("Trimesh thickness error: ", len(hits), " hits detected. ", len(origins), "hits expected.")
    #     return
    distances = np.linalg.norm(hits - hit_origins, axis=1)
    wall_thicknesses = distances

    # Now normalize
    wall_thicknesses -= np.amin(wall_thicknesses)
    max_thickness = np.amax(wall_thicknesses)
    wall_thicknesses /= max_thickness

    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    # mesh.visual.face_colors = cmap(wall_thicknesses)
    colors = cmap(wall_thicknesses)
    colors[:, 3] = 0.8

    point_cloud = trimesh.points.PointCloud(vertices=hit_origins,
                                            colors=colors)

    s = trimesh.Scene()
    s.add_geometry(point_cloud)
    s.add_geometry(mesh)
    s.show()

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

    s = trimesh.Scene()

    gap_sizes -= np.amin(gap_sizes)
    max_thickness = np.amax(gap_sizes)
    gap_sizes /= max_thickness

    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    colors = 255.0 * cmap(gap_sizes)
    point_cloud = trimesh.points.PointCloud(vertices=hit_origins,
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
    mesh_path = paths.get_thingiverse_stl_path(5561)
    # mesh_path = 'stls/crane.stl'
    mesh = trimesh.load(mesh_path)
    # # mesh = trimesh_util.TRIMESH_TEST_MESH
    # show_sampled_gaps(mesh)
    show_sampled_thickness(mesh)

    # # ## Multi STL
    # for i in range(20):
    #     random_index = random.randint(0, 10000)
    #     print(random_index)
    #     mesh_path = paths.get_thingiverse_stl_path(random_index)
    #     mesh = trimesh.load(mesh_path)
    #     # show_sampled_thickness(mesh)
    #     # sample_evenly_and_show(mesh)
    #     show_sampled_gaps(mesh)

