# if you have pyembree this should be decently fast
import matplotlib.pyplot as plt
import trimesh
import trimesh_util
import numpy as np
import paths
import random
import stopwatch
import pandas as pd
from tqdm import tqdm

NO_GAP_VALUE = -1

def sample_and_get_normals(mesh, mesh_aux: trimesh_util.MeshAuxilliaryInfo):
    sample_points, face_index = trimesh.sample.sample_surface_even(mesh, 50000)
    normals = mesh_aux.facet_normals[face_index]
    return sample_points, normals

def get_sampled_thickness(mesh):
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    origins, normals = sample_and_get_normals(mesh, mesh_aux)

    facet_offset = -normals * 0.001
    hits = mesh.ray.intersects_location(ray_origins=origins + facet_offset,
                                        ray_directions=-normals,
                                        multiple_hits=False)[0]

    if len(hits) != len(origins):
        print("Trimesh thickness error: ", len(hits), " hits detected. ", len(origins), "hits expected.")
        return
    distances = np.linalg.norm(hits - origins, axis=1)

    return origins, distances

def get_sampled_gaps(mesh):
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    origins, normals = sample_and_get_normals(mesh, mesh_aux)

    num_samples = len(origins)
    gap_sizes = np.empty(num_samples)

    for i in tqdm(range(num_samples)):
        normal = normals[i, :]
        origin = origins[i, :]
        facet_offset = normal * 0.001
        hits = mesh.ray.intersects_location(ray_origins=(origin + facet_offset)[np.newaxis, :],
                                                 ray_directions=normal[np.newaxis, :],
                                                 multiple_hits=False)[0]
        if len(hits) == 0:
            gap_sizes[i] = NO_GAP_VALUE
        else:
            first_hit = hits[0]
            distance = np.linalg.norm(origin - first_hit)
            gap_sizes[i] = distance

    print("Num gap samples: ", len(gap_sizes[gap_sizes != NO_GAP_VALUE]))

    # Now normalize
    if len(gap_sizes[gap_sizes != NO_GAP_VALUE]) > 0:
        gap_sizes[gap_sizes != NO_GAP_VALUE] -= np.amin(gap_sizes[gap_sizes != NO_GAP_VALUE])
        max_thickness = np.amax(gap_sizes)
        gap_sizes[gap_sizes != NO_GAP_VALUE] /= max_thickness

        gap_points = origins[gap_sizes != NO_GAP_VALUE]
        gaps = gap_sizes[gap_sizes != NO_GAP_VALUE]

        return gap_points, gaps

def save(points, values, filename):
    names = ["x", "y", "z", "value"]
    data = np.concatenate([points, values[:, np.newaxis]], axis=1)
    df = pd.DataFrame(data=data, columns=names)
    print("saving")
    print(filename)
    df.to_csv(filename + ".csv", index=False, header=True)



if __name__ == "__main__":
    ## Single STL
    # mesh_path = paths.get_onshape_stl_path(185)
    # mesh_path = paths.get_thingiverse_stl_path(1324)
    # mesh_path = 'stls/crane.stl'
    # mesh = trimesh.load(mesh_path)
    # # mesh = trimesh_util.TRIMESH_TEST_MESH
    # show_sampled_gaps(mesh)

    # ## Multi STL
    for i in range(20):
        # random_index = random.randint(0, 10000)
        # print(random_index)
        random_index = i
        mesh_path = paths.get_thingiverse_stl_path(random_index)
        mesh = trimesh.load(mesh_path)
        # show_sampled_thickness(mesh)
        points, values = get_sampled_gaps(mesh)
        save(points, values, paths.HOME_PATH + "generation_output/TRIMESH" + str(random_index))


