import trimesh
import trimesh_util
from trimesh_util import MeshAuxilliaryInfo
import paths
import numpy as np
import util
import pandas as pd
import printability_metrics
from tqdm import tqdm
import json

def calculate_mesh_info(mesh, mesh_save_name):
    mesh_aux = MeshAuxilliaryInfo(mesh)

    # Only take the largest body in a mesh
    if mesh.body_count > 1:
        splits = list(mesh.split(only_watertight=False))
        largest_volume = 0
        largest_submesh = None
        for submesh in splits:
            temp_volume = submesh.volume
            if temp_volume > largest_volume:
                largest_volume = temp_volume
                largest_submesh = submesh
        mesh = largest_submesh
        mesh_aux = MeshAuxilliaryInfo(largest_submesh)

    data = {
        "vertices": mesh_aux.num_vertices,
        "edges": mesh_aux.num_edges,
        "faces": mesh_aux.num_facets,
        "scale": np.max(mesh_aux.bound_length),
        "num_objects": mesh.body_count,
        "volume": mesh.volume,
        "overhang_violation": printability_metrics.get_overhang_printability(mesh_aux)[2],
        "stairstep_violation": printability_metrics.get_stairstep_printability(mesh_aux)[2],
        "thickness_violation": printability_metrics.get_thickness_printability(mesh_aux)[2],
        "gap_violation": printability_metrics.get_gap_printability(mesh_aux)[2],
        "orientation": {
            "r": 0.0,
            "p": 0.0,
            "y": 0.0
        },
        "mesh_location": mesh_save_name
    }
    return data, mesh

if __name__ == "__main__":
    # ## Multi STL
    # Collects statistics along given range
    use_onshape = True
    data_path = paths.HOME_PATH + "data/"

    if use_onshape:
        max_range = 290
        mesh_scale = 25.4
        prefix = "onshape"
    else:
        max_range = 400
        mesh_scale = 1.0
        prefix = "thing"

    for i in tqdm(range(max_range)):
        index = i
        if use_onshape:
            mesh_path = paths.get_onshape_stl_path(index, get_by_order=True)
        else:
            mesh_path = paths.get_thingiverse_stl_path(index)

        base_name = prefix + "_" + "mesh" + str(i)
        save_data_path = data_path + base_name + ".json"
        save_mesh_path = data_path + "mesh/" + base_name + ".stl"

        mesh = trimesh.load(mesh_path, force='mesh')

        if not trimesh_util.mesh_is_valid(mesh):
            print("mesh skipped:", i)
            continue

        mesh.apply_scale(mesh_scale)
        mesh_info, mesh_to_save = calculate_mesh_info(mesh, save_mesh_path)
        mesh_to_save.export(save_mesh_path, file_type="stl")
        with open(save_data_path, 'w') as f:
            json.dump(mesh_info, f)


