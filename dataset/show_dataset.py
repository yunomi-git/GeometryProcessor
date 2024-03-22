import trimesh
import trimesh_util
from trimesh_util import MeshAuxilliaryInfo
import paths
import random
import numpy as np
from process_and_save import MeshDatasetFileManager
import util

from process_and_save import calculate_instance_target, get_augmented_mesh

if __name__ == "__main__":
    mode = "single" # single or multi
    if mode == "single":
        # Single STL
        # mesh_path = paths.get_onshape_stl_path(233)
        # mesh_path = paths.get_thingiverse_stl_path(258, get_by_order=True)
        mesh_path = paths.HOME_PATH + 'stls/low-res.stl'
        mesh = trimesh.load(mesh_path)
        # trimesh_util.show_mesh(mesh)
        #
        # augmentations = {
        #     "orientation": {
        #         "x": 0.0,
        #         "y": np.pi/2,
        #         "z": 0.0
        #     },
        #     "scale": 1
        # }
        #
        # mesh = get_augmented_mesh(mesh, augmentations)
        # trimesh_util.show_mesh(mesh)

        ## Normals
        # mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
        # points, normals = mesh_aux.sample_and_get_normals(count=5000, even=False, area_weight=False)
        # trimesh_util.show_mesh_with_normals(mesh, points, normals)

        ## Samples
        mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
        points, values = mesh_aux.calculate_curvature_samples(use_gaussian=True, count=10000)
        # points, values = mesh_aux.calculate_gap_samples()

        # points, values = mesh_aux.calculate_thicknesses_samples()
        # characteristic_length = np.mean(mesh_aux.bound_length) / 20.0
        # points[values > characteristic_length] = trimesh_util.NO_GAP_VALUE

        # points, values = mesh_aux.calculate_overhangs_samples()
        #
        trimesh_util.show_sampled_values(mesh, points, values)
    else:
        file_manager = MeshDatasetFileManager(paths.HOME_PATH + "data2/")
        mesh_paths = file_manager.get_mesh_files(absolute=True)
        num_meshes = len(mesh_paths)
        timer = util.Stopwatch()
        ## Multi STL
        for i in range(num_meshes):
            mesh_path = mesh_paths[i]
            print(mesh_path)

            timer.start()
            mesh = trimesh.load(mesh_path)
            timer.print_time("load mesh")

            timer.start()
            mesh_aux = MeshAuxilliaryInfo(mesh)
            timer.print_time("calc mesh aux")

            ## Transformation timing
            # timer.start()
            # mesh_aux = mesh_aux.get_transformed_mesh(scale=2.0, orientation=[np.pi / 3, np.pi/4, np.pi/4])
            # timer.print_time("transform")
            # mesh = mesh_aux.mesh
            #
            # timer.start()
            # outputs = calculate_instance_outputs(mesh, augmentations={})
            # timer.print_time("calculate outputs")
            ## End Transformation Timing

            # trimesh_util.show_mesh(mesh)
            # trimesh_util.show_mesh_with_orientation(mesh)

            ## Samples
            # points, values = mesh_aux.calculate_gap_samples()
            # points, values = mesh_aux.calculate_thicknesses_samples()
            points, values = mesh_aux.calculate_overhangs_samples(cutoff_angle_rad=np.pi/4)
            trimesh_util.show_sampled_values(mesh, points, values)

            ## Facets
            # values = mesh_aux.calculate_gap_facets()
            # values = mesh_aux.calculate_thicknesses_facets()
            # trimesh_util.show_mesh_with_facet_colors(mesh, values)

