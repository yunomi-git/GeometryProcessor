import trimesh

import paths
import trimesh_util
import numpy as np
from pytorch3d.structures.meshes import Meshes
import torch
import torch.nn.functional as f
# Referencs: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_normal_consistency.html

def get_threshold_penalty(x_warn, x_fail, crossover=0.05):
    # Function is 1/ 1 + exp(-a (x + b)). We need to solve for a and b
    cross_warn = np.log(1.0 / crossover - 1)
    cross_fail = np.log(1.0 / (1 - crossover) - 1)
    a = (cross_warn - cross_fail) / (x_fail - x_warn)
    b = (cross_warn * x_fail - x_fail * x_warn) / (cross_fail - cross_warn)
    penalty_func = lambda x: 1.0 / (1 + torch.exp(-a * (x + b)))
    return penalty_func

def calculate_normals(mesh: trimesh.Trimesh):
    # mesh: N faces Nx3
    #       M vertices Mx3
    # outputs: Vertex normals Mx3
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    vertices = mesh_aux.vertices
    face_indices = mesh_aux.facets

    v0 = vertices[face_indices[:, 0]]
    v1 = vertices[face_indices[:, 1]]
    v2 = vertices[face_indices[:, 2]]

    n0 = np.cross(v1-v0, v0-v2, axis=1)
    # n1 = np.cross(v2 - v1, v1 - v0, axis=1)
    # n2 = np.cross(v0 - v2, v2 - v1, axis=1)

    areas = np.linalg.norm(n0, axis=1) / 2
    normals = -n0 / np.linalg.norm(n0)
    # These are the normals
    # Convert to angle

    return normals

def calculate_normals_differentiable(meshes: Meshes):
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    # edges_packed = meshes.edges_packed()  # (sum(E_n), 2)
    verts_packed_to_mesh_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)

    v0 = verts_packed[faces_packed[:, 0]]
    v1 = verts_packed[faces_packed[:, 1]]
    v2 = verts_packed[faces_packed[:, 2]]

    n0 = torch.cross(v1-v0, v0-v2, dim=1)
    normals = -f.normalize(n0, p=2, dim=1)
    areas = torch.linalg.vector_norm(n0, dim=1)
    return normals, areas


    # loss = 1 - torch.cosine_similarity(n0, n1, dim=1)

    # verts_packed_to_mesh_idx = verts_packed_to_mesh_idx[vert_idx[:, 0]]
    # verts_packed_to_mesh_idx = verts_packed_to_mesh_idx[vert_edge_pair_idx[:, 0]]
    # num_normals = verts_packed_to_mesh_idx.bincount(minlength=N)
    # weights = 1.0 / num_normals[verts_packed_to_mesh_idx].float()

    # loss = loss * weights
    # return loss.sum() / N

def normals_to_cost_differentiable(normals):
    # Need to normalize the normals

    normals_z = normals[:, 2]
    angles = torch.arcsin(normals_z)  # arcsin calculates overhang angles as < 0
    # samples_above_floor = vertices[:, 2] >= (layer_height + self.bound_lower[2])
    # angles = angles[samples_above_floor]
    loss_func = get_threshold_penalty(x_warn=-torch.pi/4, x_fail=-torch.pi/2, crossover=0.05)
    loss = loss_func(angles)
    loss = torch.sum(loss)
    return loss


if __name__=="__main__":
    mesh = trimesh.load(paths.HOME_PATH + "stls/low-res.stl")
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    # normals = calculate_normals(mesh)

    #convert to tensor
    with torch.enable_grad():
        tensor_verts = torch.from_numpy(mesh_aux.vertices).float()
        tensor_faces = torch.from_numpy(mesh_aux.facets).float()
        model = torch.nn.Linear(3,3)
        with torch.no_grad():
            model.weight.copy_(torch.eye(3))
            model.bias.copy_(torch.zeros(3))
        out_verts = model(tensor_verts)
        meshes = Meshes(faces=[tensor_faces], verts=[out_verts])
        normals_tensor, areas_tensor = calculate_normals_differentiable(meshes)
        loss = normals_to_cost_differentiable(normals_tensor)
        loss.backward()
    # normals = normals_tensor.numpy()

    # face_centroids = mesh_aux.facet_centroids
    # trimesh_util.show_mesh_with_normals(mesh, face_centroids, normals)
