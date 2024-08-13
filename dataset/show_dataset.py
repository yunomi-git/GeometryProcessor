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
        mesh_path = paths.HOME_PATH + 'stls/Octocat.stl'
        # mesh_path = paths.RAW_DATASETS_PATH + 'DrivAerNet/Simplified_Remesh/N_S_WW_WM_3/Exp_001/N_S_WW_WM_3.stl'
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
        # mesh = trimesh_util.get_transformed_mesh(mesh, orientation=np.array([np.pi/4, np.pi/3, 0]))
        # trimesh_util.show_mesh_with_z_normal(mesh)
        # trimesh_util.show_mesh(mesh)

        # mesh = trimesh_util.get_transformed_mesh(mesh, scale=1.0, translation=np.array([1, 0, 0]),
        #                                          orientation=np.array([0.0, 0.0, np.pi/2]))
        # mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
        # centroid = np.mean(mesh_aux.vertices, axis=0)
        # print(centroid)
        trimesh_util.show_mesh(mesh)
        trimesh_util.show_mesh(mesh, isometric=True)

        ## Normals
        # mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
        # points, normals = mesh_aux.sample_and_get_normals(count=5000, use_weight='even')
        # samples = mesh_aux.vertices
        # normals = mesh_aux.vertex_normals
        # trimesh_util.show_mesh_with_normals(mesh, samples, normals)

        ## Samples
        # mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
        # points, values = mesh_aux.calculate_curvature_samples(curvature_method="abs", count=5000)
        # samples, normals = mesh_aux.get_vertices_and_normals()
        # points, values = mesh_aux.calculate_thickness_at_points(samples, normals, return_num_samples=False)

        # points, values, vertex_ids = mesh_aux.calculate_thickness_at_points(samples, normals, return_num_samples=False, return_ray_ids=True)
        # print("Num Inputs", len(samples))
        # print("Num Outputs", len(points))
        # values = trimesh_util.repair_missing_mesh_values(mesh, vertex_ids=vertex_ids, values=values, max_iterations=2)
        # points = mesh_aux.vertices

        # points, values = mesh_aux.calculate_gap_samples()

        # points, values = mesh_aux.calculate_thicknesses_samples()
        # characteristic_length = np.mean(mesh_aux.bound_length) / 20.0
        # points[values > characteristic_length] = trimesh_util.NO_GAP_VALUE

        # points, values = mesh_aux.calculate_overhangs_samples()
        #
        # trimesh_util.show_sampled_values(mesh, points, values)
    else:
        file_manager = MeshDatasetFileManager(paths.CACHED_DATASETS_PATH + "data_th5k_norm/")
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

            # mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
            # points, normals = mesh_aux.sample_and_get_normals(count=5000, even=False, area_weight=False)
            points = mesh_aux.vertices
            normals = mesh_aux.vertex_normals
            trimesh_util.show_mesh_with_normals(mesh, points, normals)

            ## Samples
            # points, values = mesh_aux.calculate_gap_samples()
            # points, values = mesh_aux.calculate_thicknesses_samples()
            # points, values = mesh_aux.calculate_overhangs_samples(cutoff_angle_rad=np.pi/4)
            # trimesh_util.show_sampled_values(mesh, points, values)

            ## Facets
            # values = mesh_aux.calculate_gap_facets()
            # values = mesh_aux.calculate_thicknesses_facets()
            # trimesh_util.show_mesh_with_facet_colors(mesh, values)

