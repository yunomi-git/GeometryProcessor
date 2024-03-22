import trimesh
import trimesh_util
from trimesh_util import MeshAuxilliaryInfo
import paths
import random
import numpy as np
from dataset.process_and_save import MeshDatasetFileManager
import util

from dataset.process_and_save import calculate_instance_target, get_augmented_mesh

if __name__ == "__main__":
    mode = "multi" # single or multi
    meshes_to_try = [
                     paths.HOME_PATH + 'stls/hi-res.stl',
                     paths.HOME_PATH + 'stls/Hub.stl',
                     paths.HOME_PATH + 'stls/crane.stl',
                     paths.get_onshape_stl_path(253)
                     ]

        ## Multi STL
    for file in meshes_to_try:
        mesh = trimesh.load(file)
        mesh_aux = MeshAuxilliaryInfo(mesh)
        # points, values = mesh_aux.calculate_curvature_samples(curvature_method="defect", count=4096, sampling_method="mixed")
        # points, values = mesh_aux.calculate_surface_defect_vertices()
        # values = np.log(np.abs(values))

        points, normals = mesh_aux.sample_and_get_normals(count=4096, use_weight="mixed")
        points, values = mesh_aux.calculate_thickness_at_points(points, normals, return_num_samples=False)

        trimesh_util.show_sampled_values(mesh, points, values)

        # values = mesh_aux.calculate_surface_defects_facets()
        # log_values = np.log(np.abs(values))
        # trimesh_util.show_mesh_with_facet_colors(mesh, log_values)





