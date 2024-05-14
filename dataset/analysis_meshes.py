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
from tqdm import tqdm


# Issues: onshape 202, thing 285, onshape 37, onehape 89
if __name__ == "__main__":
    # ## Multi STL
    # Collects statistics along given range

    metrics = util.DictionaryList()

    use_onshape = False
    if use_onshape:
        max_range = 290
        mesh_scale = 25.4
    else:
        max_range = 400
        mesh_scale = 1.0

    for i in tqdm(range(max_range)):
        # if i == 175:
        #     print("hi")
        index = i
        # print(i)
        if use_onshape:
            mesh_path = paths.get_onshape_stl_path(index, get_by_order=True)
        else:
            mesh_path = paths.get_thingiverse_stl_path(index)

        mesh = trimesh.load(mesh_path, force='mesh')

        if trimesh_util.mesh_is_valid(mesh):
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
            if not mesh.is_watertight:
                trimesh.repair.broken_faces(mesh, (255, 0, 0, 255))
                trimesh_util.show_mesh(mesh)
                trimesh.repair.fill_holes(mesh)
                if not mesh.is_watertight:
                    print("not fixed")



            metrics.add_element({
                "vertices": mesh_aux.num_vertices,
                "edges": mesh_aux.num_edges,
                "faces": mesh_aux.num_faces,
                "scale": np.max(mesh_aux.bound_length),
                "num_objects": mesh.body_count,
                "volume": mesh.volume,
                "manifold": int(mesh.is_watertight) + 1
            })

    # display as histogram
    # data = pd.DataFrame(data={
    #     "vertices": vertices,
    #     "edges": edges,
    #     "faces": faces,
    #     "scale": scale,
    #     "num_objects": num_objects,
    #     "volume": volume
    # })
    data = pd.DataFrame(data=metrics.master_list)
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
    sns.histplot(data=data, x="manifold", log_scale=True)
    plt.show()