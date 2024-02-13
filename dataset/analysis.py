import trimesh
import trimesh_util
from trimesh_util import MeshAuxilliaryInfo
import paths
import random
import numpy as np
import util

if __name__ == "__main__":
    # ## Multi STL
    # Collects statistics along given range
    max_range = 300

    vertices = np.empty(max_range)
    edges = np.empty(max_range)
    faces = np.empty(max_range)
    scale = np.empty(max_range)

    for i in range(300):
        # random_index = random.randint(0, 10000)
        # print(random_index)
        # mesh_path = paths.get_thingiverse_stl_path(random_index)

        # index = random.randint(0, 300)
        index = i
        mesh_path = paths.get_onshape_stl_path(index)

        mesh = trimesh.load(mesh_path)
        mesh_aux = MeshAuxilliaryInfo(mesh)

        vertices[i] = mesh_aux.num_vertices
        edges[i] = mesh_aux.num_edges
        faces[i] = mesh_aux.num_facets
        scale[i] = np.max(mesh_aux.bound_length)

    # display as histogram
