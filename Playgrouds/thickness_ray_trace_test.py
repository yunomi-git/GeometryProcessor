# if you have pyembree this should be decently fast
import matplotlib.pyplot as plt
import trimesh
import trimesh_util
import numpy as np
import paths
import random
import stopwatch

def calculate_and_show_thickness(mesh):
    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    num_facets = mesh_aux.num_faces

    is_thin_vector = np.empty(num_facets)
    facet_offset = -mesh_aux.face_normals * 0.001
    hits = mesh.ray.intersects_location(ray_origins=mesh_aux.face_centroids + facet_offset,
                                        ray_directions=-mesh_aux.face_normals,
                                        multiple_hits=False)[0]

    if len(hits) != num_facets:
        return
    distances = np.linalg.norm(hits - mesh_aux.face_centroids, axis=1)
    wall_thicknesses = distances

    # Now normalize
    wall_thicknesses -= np.amin(wall_thicknesses)
    max_thickness = np.amax(wall_thicknesses)
    wall_thicknesses /= max_thickness

    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    mesh.visual.face_colors = cmap(wall_thicknesses)

    s = trimesh.Scene()
    s.add_geometry(mesh)
    s.show()

if __name__ == "__main__":
    # mesh_path = paths.get_onshape_stl_path(2)
    # mesh_path = 'stls/crane.stl'
    # mesh = trimesh.load(mesh_path)
    # mesh = trimesh_util.TRIMESH_TEST_MESH

    for i in range(20):
        mesh_path = paths.get_onshape_stl_path(random.randint(1, 300))
        mesh = trimesh.load(mesh_path)
        calculate_and_show_thickness(mesh)
