from Generation import CalculateNormalsPlaygraound as normals
import trimesh_util
import trimesh
import paths
import numpy as np
from pytorch3d.structures.meshes import Meshes
import torch
import torch.nn.functional as f
import torch.optim as optim
import math
from torch.autograd import Variable
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
# https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh
torch.autograd.set_detect_anomaly(True)
def show_mesh(meshes: Meshes):
    optimized_mesh = trimesh.Trimesh(vertices=meshes.verts_packed().detach().cpu().numpy(),
                           faces=meshes.faces_packed().detach().cpu().numpy())
    trimesh_util.show_mesh_with_z_normal(optimized_mesh)

if __name__ == "__main__":
    mesh = trimesh.load(paths.get_thingiverse_stl_path(258, get_by_order=True))
    # mesh = trimesh.load(paths.HOME_PATH + "stls/low-res.stl")
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    # First scale mesh down
    centroid = np.mean(mesh_aux.vertices, axis=0)
    min_bounds = mesh_aux.bound_lower
    normalization_translation = -np.array([centroid[0], centroid[1], min_bounds[2]])
    scale = max(mesh_aux.bound_length)
    normalization_scale = 1.0 / scale
    # Orient if needed
    orientation = np.array([0, np.pi/2, 0])
    # orientation = np.array([0, 0.0, 0])
    mesh = trimesh_util.get_transformed_mesh(mesh, scale=normalization_scale,
                                             translation=normalization_translation,
                                             orientation=orientation)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    device = torch.device("cuda")

    tensor_verts = torch.from_numpy(mesh_aux.vertices).float().to(device)
    tensor_faces = torch.from_numpy(mesh_aux.facets).float().to(device)
    meshes = Meshes(faces=[tensor_faces], verts=[tensor_verts])
    # tensor_verts = Variable(tensor_verts, requires_grad=True)

    deform_verts = torch.full(meshes.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = optim.Adam([deform_verts], lr=1e-4)

    plot_every_num = 5

    for i in range(200):
        print("====== iter", i, "===========")
        # Initialize optimizer
        optimizer.zero_grad()

        # Deform the mesh
        new_mesh = meshes.offset_verts(deform_verts)
        if torch.isnan(new_mesh.verts_packed()).any():
            print("nan")

        smoothing_loss = mesh_laplacian_smoothing(new_mesh, method="uniform") #1e-2
        loss_normal = mesh_normal_consistency(new_mesh) # 1e-1
        overhang_loss = normals.loss_mesh_overhangs(new_mesh) # 1e-4
        stairstep_loss = normals.loss_mesh_stairsteps(new_mesh) # 1e-4

        loss = (
                overhang_loss +
                stairstep_loss +
                # 1e-5 * smoothing_loss +
                1e-4 * loss_normal
        )
        if math.isnan(loss.item()):
            print("nan")
        print("overhang", overhang_loss.item(), "stair", stairstep_loss.item(), "\nnormal", loss_normal.item(), "smooth", smoothing_loss.item())
        loss.backward()

        if torch.isnan(deform_verts.grad).any():
            print("grad nan")


        # Print the losses
        print('total_loss = %.6f' % loss)
        torch.nn.utils.clip_grad_norm_(deform_verts, 1)
        # Optimization step
        optimizer.step()

        # if i % plot_every_num == 0:
        #     show_mesh(new_mesh)

    # Visualize
    optimized_mesh = trimesh.Trimesh(vertices=new_mesh.verts_packed().detach().cpu().numpy(),
                           faces=new_mesh.faces_packed().detach().cpu().numpy())
    trimesh_util.show_mesh_with_z_normal(mesh)
    trimesh_util.show_mesh_with_z_normal(optimized_mesh)



    # Update
