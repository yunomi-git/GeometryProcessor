import trimesh
import trimesh_util
from trimesh_util import MeshAuxilliaryInfo
import paths
import random

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
        # random_index = random.randint(0, 10000)
        # print(random_index)
        # mesh_path = paths.get_thingiverse_stl_path(random_index)

        random_index = random.randint(0, 300)
        print(random_index)
        mesh_path = paths.get_onshape_stl_path(random_index)

        mesh = trimesh.load(mesh_path)
        mesh_aux = MeshAuxilliaryInfo(mesh)

        trimesh_util.show_mesh_with_orientation(mesh)

        ## Samples
        # points, values = mesh_aux.calculate_gap_samples()
        # points, values = mesh_aux.calculate_thicknesses_facets()
        # trimesh_util.show_sampled_values(mesh, points, values)

        ## Facets
        # values = mesh_aux.calculate_gap_facets()
        # values = mesh_aux.calculate_thicknesses_facets()
        # trimesh_util.show_mesh_with_facet_colors(mesh, values)

