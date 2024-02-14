import trimesh
import trimesh_util
from trimesh_util import MeshAuxilliaryInfo
import paths
import random
import numpy as np
import util
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # ## Multi STL
    # Collects statistics along given range
    use_onshape = False
    if use_onshape:
        max_range = 290
        mesh_scale = 25.4
    else:
        max_range = 400
        mesh_scale = 1.0

    vertices = np.empty(max_range)
    edges = np.empty(max_range)
    faces = np.empty(max_range)
    scale = np.empty(max_range)
    num_objects = np.empty(max_range)
    volume = np.empty(max_range)

    for i in range(max_range):
        # if i == 175:
        #     print("hi")
        index = i
        print(i)
        if use_onshape:
            mesh_path = paths.get_onshape_stl_path(index, get_by_order=True)
        else:
            mesh_path = paths.get_thingiverse_stl_path(index)

        mesh = trimesh.load(mesh_path, force='mesh')

        if not trimesh_util.mesh_is_valid(mesh):
            vertices[i] = 0
            edges[i] = 0
            faces[i] = 0
            scale[i] = 0
            num_objects[i] = 0
            volume[i] = 0
        else:
            mesh.apply_scale(mesh_scale)
            mesh_aux = MeshAuxilliaryInfo(mesh)
            # pause if number of bodies is > 1
            if mesh.body_count > 1:
                splits = list(mesh.split(only_watertight=False))
                largest_volume = 0
                largest_submesh = None
                for submesh in splits:
                    temp_volume = submesh.volume
                    if temp_volume > largest_volume:
                        largest_volume = temp_volume
                        largest_submesh = submesh
                # trimesh_util.show_mesh(largest_submesh)
                mesh = largest_submesh
                mesh_aux = MeshAuxilliaryInfo(largest_submesh)

            vertices[i] = mesh_aux.num_vertices
            edges[i] = mesh_aux.num_edges
            faces[i] = mesh_aux.num_facets
            scale[i] = np.max(mesh_aux.bound_length)
            num_objects[i] = mesh.body_count
            volume[i] = mesh.volume

    # display as histogram
    data = pd.DataFrame(data={
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "scale": scale,
        "num_objects": num_objects,
        "volume": volume
    })
    sns.histplot(data=data, x="vertices", log_scale=True)
    plt.show()
    sns.histplot(data=data, x="edges", log_scale=True)
    plt.show()
    sns.histplot(data=data, x="faces", log_scale=True)
    plt.show()
    sns.histplot(data=data, x="scale", log_scale=True)
    plt.show()
    sns.histplot(data=data, x="num_objects", log_scale=True)
    plt.show()
    sns.histplot(data=data, x="volume", log_scale=True)
    plt.show()