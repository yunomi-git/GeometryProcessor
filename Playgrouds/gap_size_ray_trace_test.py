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

def calculate_and_show_gap(mesh):
    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    num_facets = mesh_aux.num_faces
    gap_sizes = np.empty(num_facets)

    for i in tqdm(range(num_facets)):
        normal = mesh_aux.face_normals[i, :]
        origin = mesh_aux.face_centroids[i, :]
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

    # Now normalize
    if len(gap_sizes[gap_sizes != NO_GAP_VALUE]) > 0:
        gap_sizes[gap_sizes != NO_GAP_VALUE] -= np.amin(gap_sizes[gap_sizes != NO_GAP_VALUE])
        max_thickness = np.amax(gap_sizes)
        gap_sizes[gap_sizes != NO_GAP_VALUE] /= max_thickness

    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    no_gap_color = np.array([100, 100, 100, 255])
    mesh.visual.face_colors = cmap(gap_sizes)
    mesh.visual.face_colors[gap_sizes == NO_GAP_VALUE] = no_gap_color


    s = trimesh.Scene()
    s.add_geometry(mesh)
    s.show()

if __name__ == "__main__":
    ## Single STL
    mesh_path = paths.get_onshape_stl_path(165)
    # mesh_path = 'stls/crane.stl'
    mesh = trimesh.load(mesh_path)
    # mesh = trimesh_util.TRIMESH_TEST_MESH
    calculate_and_show_gap(mesh)

    ## Multi STL
    # for i in range(20):
    #     mesh_path = paths.get_onshape_stl_path(random.randint(1, 300))
    #     mesh = trimesh.load(mesh_path)
    #     calculate_and_show_gap(mesh)
