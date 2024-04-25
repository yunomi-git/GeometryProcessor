import trimesh
import trimesh_util
import numpy as np
import paths


def normalize_mesh(mesh: trimesh.Trimesh, center, normalize_scale):
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    normalization_scale = 1.0
    normalization_translation = np.array([0, 0, 0])
    if center:
        centroid = np.mean(mesh_aux.vertices, axis=0)
        min_bounds = mesh_aux.bound_lower
        normalization_translation = -np.array([centroid[0], centroid[1], min_bounds[2]])
    if normalize_scale:
        scale = max(mesh_aux.bound_length)
        normalization_scale = 1.0 / scale

    mesh = trimesh_util.get_transformed_mesh_trs(mesh, scale=normalization_scale, translation=normalization_translation)
    return mesh

if __name__=="__main__":
    mesh = trimesh.load(paths.HOME_PATH + "stls/low-res.stl")
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    print("lb", mesh_aux.bound_lower)
    print("ub", mesh_aux.bound_upper)
    print("length", mesh_aux.bound_length)

    mesh = normalize_mesh(mesh, True, True)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    print("lb", mesh_aux.bound_lower)
    print("ub", mesh_aux.bound_upper)
    print("length", mesh_aux.bound_length)