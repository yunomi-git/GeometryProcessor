import trimesh
import trimesh_util
from trimesh_util import MeshAuxilliaryInfo
import paths
import random
import numpy as np
from process_and_save import MeshDatasetFileManager
import util

from process_and_save import calculate_instance_outputs

if __name__ == "__main__":
    # # Single STL
    # # mesh_path = paths.get_onshape_stl_path(233)
    # # mesh_path = paths.get_thingiverse_stl_path(258, get_by_order=True)
    # mesh_path = paths.HOME_PATH + 'stls/crane.stl'
    # mesh = trimesh.load(mesh_path)
    # # mesh = trimesh_util.TRIMESH_TEST_MESH
    # trimesh_util.show_mesh(mesh)
    #
    #
    # ## Samples
    # mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    # # points, values = mesh_aux.calculate_gap_samples()
    #
    # points, values = mesh_aux.calculate_thicknesses_samples()
    # characteristic_length = np.mean(mesh_aux.bound_length) / 20.0
    # points[values > characteristic_length] = trimesh_util.NO_GAP_VALUE
    #
    # trimesh_util.show_sampled_values(mesh, points, values)

    file_manager = MeshDatasetFileManager(paths.HOME_PATH + "data2/")
    mesh_paths = file_manager.get_mesh_files(absolute=True)
    num_meshes = len(mesh_paths)
    timer = util.Stopwatch()
    ## Multi STL
    for i in range(num_meshes):
        # random_index = random.randint(0, 10000)
        # print(random_index)
        # mesh_path = paths.get_thingiverse_stl_path(random_index)

        # random_index = random.randint(0, 300)
        # print(random_index)
        # mesh_path = paths.get_onshape_stl_path(random_index)
        mesh_path = mesh_paths[i]
        print(mesh_path)

        timer.start()
        mesh = trimesh.load(mesh_path)
        timer.print_time("load")

        timer.start()
        mesh_aux = MeshAuxilliaryInfo(mesh)
        timer.print_time("calc aux")

        timer.start()
        mesh_aux = mesh_aux.get_transformed_mesh(scale=2.0, orientation=[np.pi / 3, np.pi/4, np.pi/4])
        timer.print_time("transform")
        mesh = mesh_aux.mesh

        timer.start()
        outputs = calculate_instance_outputs(mesh, augmentations={})
        timer.print_time("calculate outputs")

        # trimesh_util.show_mesh(mesh)
        trimesh_util.show_mesh_with_orientation(mesh)

        ## Samples
        # points, values = mesh_aux.calculate_gap_samples()
        # points, values = mesh_aux.calculate_thicknesses_samples()
        # trimesh_util.show_sampled_values(mesh, points, values)

        ## Facets
        # values = mesh_aux.calculate_gap_facets()
        # values = mesh_aux.calculate_thicknesses_facets()
        # trimesh_util.show_mesh_with_facet_colors(mesh, values)

