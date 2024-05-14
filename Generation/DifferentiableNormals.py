import trimesh

import paths
import trimesh_util
import numpy as np
from pytorch3d.structures.meshes import Meshes
import torch
import torch.nn.functional as f

from Generation.Thresholds import get_threshold_penalty

# Referencs: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_normal_consistency.html

torch.cuda.empty_cache()


def calculate_normals(mesh: trimesh.Trimesh):
    # mesh: N faces Nx3
    #       M vertices Mx3
    # outputs: Vertex normals Mx3
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    vertices = mesh_aux.vertices
    face_indices = mesh_aux.faces

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

    # print("v0", v0)
    # print("v1", v1)
    # print("v2", v2)
    print("v1-v0", v1-v0)
    print("v0-v2", v0-v2)

    if (torch.isnan(verts_packed).any()):
        print("nan verts packed")
    if (torch.isnan(v0).any()):
        print("nan v0 packed")
    if (torch.isnan(v1).any()):
        print("nan v1 packed")
    if (torch.isnan(v2).any()):
        print("nan v2 packed")

    n0 = torch.linalg.cross(v1-v0, v0-v2, dim=1)
    print("n0", n0)
    if (torch.isnan(n0).any()):
        print("normals nan")
    if (torch.eq(torch.norm(n0), 0).any()):
        print(torch.where(torch.eq(torch.norm(n0), 0)))
    normals = -f.normalize(n0, p=2, dim=1)
    if (torch.isnan(normals).any()):
        print("normals nan")

    areas = torch.linalg.vector_norm(n0, dim=1)
    return normals, areas

def calculate_centroids_differentiable(meshes: Meshes):
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)

    v0 = verts_packed[faces_packed[:, 0]]
    v1 = verts_packed[faces_packed[:, 1]]
    v2 = verts_packed[faces_packed[:, 2]]

    return (v0 + v1 + v2) / 3.0

def normals_to_cost_differentiable(normals):
    # Need to normalize the normals
    normals_z = normals[:, 2]
    angles = torch.arcsin(normals_z)  # arcsin calculates overhang angles as < 0
    # samples_above_floor = vertices[:, 2] >= (layer_height + self.bound_lower[2])
    # angles = angles[samples_above_floor]
    loss_func = get_threshold_penalty(x_warn=-torch.pi / 4, x_fail=-torch.pi / 2, crossover=0.05)
    loss = loss_func(angles)
    loss = torch.mean(loss)
    return loss

def loss_mesh_overhangs(meshes: Meshes):
    print("verts", meshes.verts_packed())
    normals, _ = calculate_normals_differentiable(meshes)
    normals_z = normals[:, 2]
    normals_z[normals_z > 1] = 1
    normals_z[normals_z < -1] = -1
    print("nz", normals_z)
    angles = torch.arcsin(normals_z)  # arcsin calculates overhang angles as < 0
    print("angles", angles)
    layer_height = 0.04

    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)


    v0 = verts_packed[faces_packed[:, 0]]
    v1 = verts_packed[faces_packed[:, 1]]
    v2 = verts_packed[faces_packed[:, 2]]

    min_height = torch.min(verts_packed[:, 2])

    loss_func = get_threshold_penalty(x_warn=-torch.pi / 4, x_fail=-torch.pi / 2, crossover=0.05)
    overhang_loss = loss_func(angles)
    print("loss", overhang_loss)
    if (torch.isnan(overhang_loss).any()):
        print("overhang_loss nan")

    # Create another penalty based on floor height
    centroids = (v0 + v1 + v2) / 3.0
    floor_height_loss_func = get_threshold_penalty(x_warn=min_height, x_fail=min_height + layer_height, crossover=0.05)
    floor_loss = floor_height_loss_func(centroids[:, 2])
    # floor_loss = floor_height_loss_func(v0[:, 2]) * floor_height_loss_func(v1[:, 2]) * floor_height_loss_func(v2[:, 2])
    # floor_loss = (floor_height_loss_func(v0[:, 2]) + floor_height_loss_func(v1[:, 2]) + floor_height_loss_func(v2[:, 2])) / 3.0
    if (torch.isnan(floor_loss).any()):
        print("floor_loss nan")

    # loss = floor_loss
    # loss = floor_loss * overhang_loss
    loss = overhang_loss

    loss = torch.mean(loss)
    return loss

def loss_mesh_stairsteps(meshes: Meshes):
    normals, _ = calculate_normals_differentiable(meshes)
    normals_z = normals[:, 2]
    normals_z[normals_z > 1] = 1
    normals_z[normals_z < -1] = -1
    angles = torch.arcsin(normals_z)  # arcsin calculates overhang angles as < 0

    # This one punishes high angles
    loss_func = get_threshold_penalty(x_warn=torch.pi / 4, x_fail=torch.pi / 2 * 0.90, crossover=0.05)
    # This one ignores flats
    loss_func_2 = get_threshold_penalty(x_warn=torch.pi / 2 * 0.95, x_fail=torch.pi / 2 * 0.90, crossover=0.01)
    loss = loss_func(angles) * loss_func_2(angles)
    # if (torch.isnan(loss).any()):
    #     print("loss nan")
    # loss = loss_func(angles) # TODO this in't working yet
    # loss = loss_func_2(angles)
    loss = torch.mean(loss)
    # TODO: ignore flat
    return loss






if __name__=="__main__":
    mesh = trimesh.load(paths.HOME_PATH + "stls/squirrel.stl")
    mesh = trimesh_util.normalize_mesh(mesh, center=True, normalize_scale=True)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    # normals = calculate_normals(mesh)

    #37384: squirrel
    #32770: catussy
    #39460 weird gear

    #convert to tensor
    with torch.enable_grad():
        tensor_verts = torch.from_numpy(mesh_aux.vertices).float()
        tensor_faces = torch.from_numpy(mesh_aux.faces).float()
        model = torch.nn.Linear(3,3)
        with torch.no_grad():
            model.weight.copy_(torch.eye(3))
            model.bias.copy_(torch.zeros(3))
        out_verts = model(tensor_verts)
        meshes = Meshes(faces=[tensor_faces], verts=[out_verts])
        normals_tensor, areas_tensor = calculate_normals_differentiable(meshes)
        loss = normals_to_cost_differentiable(normals_tensor)
        loss.backward()
    normals = normals_tensor.detach().numpy()

    face_centroids = mesh_aux.face_centroids
    trimesh_util.show_mesh_with_normals(mesh, face_centroids, normals * 0.2)
