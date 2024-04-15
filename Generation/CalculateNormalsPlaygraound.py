import trimesh

import paths
import trimesh_util
import numpy as np
from pytorch3d.structures.meshes import Meshes
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt
# Referencs: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_normal_consistency.html

torch.cuda.empty_cache()

def get_threshold_penalty(x_warn, x_fail, crossover=0.05):
    # Function is 1/ 1 + exp(-a (x + b)). We need to solve for a and b
    cross_warn = np.log(1.0 / crossover - 1)
    cross_fail = np.log(1.0 / (1 - crossover) - 1)
    a = (cross_warn - cross_fail) / (x_fail - x_warn)
    b = (cross_warn * x_fail - cross_fail * x_warn) / (cross_fail - cross_warn)
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

    v0 = verts_packed[faces_packed[:, 0]]
    v1 = verts_packed[faces_packed[:, 1]]
    v2 = verts_packed[faces_packed[:, 2]]

    # if (torch.isnan(verts_packed).any()):
    #     print("nan")

    n0 = torch.cross(v1-v0, v0-v2, dim=1)
    # if (torch.eq(torch.norm(n0), 0).any()):
    #     print(torch.where(torch.eq(torch.norm(n0), 0)))
    normals = -f.normalize(n0, p=2, dim=1)
    if (torch.isnan(normals).any()):
        print("normals nan")

    areas = torch.linalg.vector_norm(n0, dim=1)
    return normals, areas

def normals_to_cost_differentiable(normals):
    # Need to normalize the normals
    normals_z = normals[:, 2]
    angles = torch.arcsin(normals_z)  # arcsin calculates overhang angles as < 0
    # samples_above_floor = vertices[:, 2] >= (layer_height + self.bound_lower[2])
    # angles = angles[samples_above_floor]
    loss_func = get_threshold_penalty(x_warn=-torch.pi/4, x_fail=-torch.pi/2, crossover=0.05)
    loss = loss_func(angles)
    loss = torch.mean(loss)
    return loss

def loss_mesh_overhangs(meshes: Meshes):
    normals, _ = calculate_normals_differentiable(meshes)
    normals_z = normals[:, 2]
    angles = torch.arcsin(normals_z)  # arcsin calculates overhang angles as < 0
    layer_height = 0.04

    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)

    v0 = verts_packed[faces_packed[:, 0]]
    v1 = verts_packed[faces_packed[:, 1]]
    v2 = verts_packed[faces_packed[:, 2]]

    min_height = torch.min(verts_packed[:, 2])
    floor_height = layer_height + min_height

    # samples_above_floor = torch.gt(v0[:, 2], floor_height)
    # samples_above_floor = torch.logical_and(samples_above_floor, torch.gt(v1[:, 2], floor_height))
    # samples_above_floor = torch.logical_and(samples_above_floor, torch.gt(v2[:, 2], floor_height))

    # angles = angles[samples_above_floor]
    # print("Num points removed: ", len(faces_packed) - len(angles))
    loss_func = get_threshold_penalty(x_warn=-torch.pi / 4, x_fail=-torch.pi / 2, crossover=0.01)
    overhang_loss = loss_func(angles)

    # Create another penalty based on floor height
    floor_height_loss_func = get_threshold_penalty(x_warn=min_height, x_fail=min_height + layer_height, crossover=0.05)
    floor_loss = floor_height_loss_func(v0[:, 2]) * floor_height_loss_func(v1[:, 2]) * floor_height_loss_func(v2[:, 2])
    # floor_loss = (floor_height_loss_func(v0[:, 2]) + floor_height_loss_func(v1[:, 2]) + floor_height_loss_func(v2[:, 2])) / 3.0
    # loss = floor_loss
    loss = floor_loss * overhang_loss
    # loss = overhang_loss

    loss = torch.mean(loss)
    # TODO: ignore first layer
    return loss

def loss_mesh_stairsteps(meshes: Meshes):
    normals, _ = calculate_normals_differentiable(meshes)
    normals_z = normals[:, 2]
    angles = torch.arcsin(normals_z)  # arcsin calculates overhang angles as < 0

    # This one punishes high angles
    loss_func = get_threshold_penalty(x_warn=torch.pi / 4, x_fail=torch.pi / 2 * 0.90, crossover=0.01)
    # This one ignores flats
    loss_func_2 = get_threshold_penalty(x_warn=torch.pi/2 * 0.95, x_fail=torch.pi / 2 * 0.90, crossover=0.01)
    # loss = loss_func(angles) * loss_func_2(angles)
    # loss = loss_func(angles) # TODO this in't working yet
    loss = loss_func_2(angles)
    loss = torch.mean(loss)
    # TODO: ignore flat
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
