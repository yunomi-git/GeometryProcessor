import pyvista_util
from printability_heuristics import DifferentiableNormals as normals
import trimesh_util
import trimesh
import paths
import numpy as np
from pytorch3d.structures.meshes import Meshes
import torch
import torch.optim as optim
import math
import dataset.pymeshlab_remesh as remesh
from Generation.dgcnn_differentiable_thickness_prediction import DgcnnThickness, ThicknessLoss
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
# https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh
# torch.autograd.set_detect_anomaly(True)
def show_mesh(meshes: Meshes):
    optimized_mesh = trimesh.Trimesh(vertices=meshes.verts_packed().detach().cpu().numpy(),
                           faces=meshes.faces_packed().detach().cpu().numpy())
    # trimesh_util.show_mesh_with_z_normal(optimized_mesh)
    pyvista_util.show_mesh_z_mag(optimized_mesh)

    # pyvista_util.show_mesh(vertices=meshes.verts_packed().detach().cpu().numpy(),
    #                        faces=meshes.faces_packed().detach().cpu().numpy())

if __name__ == "__main__":
    # mesh = trimesh.load(paths.DATASETS_PATH + "Dataset_Thingiverse_10k_Remesh2/32770.stl")
    mesh = trimesh.load(paths.HOME_PATH + "stls/thickness_test_bridge.stl")
    # Normalize
    mesh = trimesh_util.normalize_mesh(mesh, center=True, normalize_scale=True)
    # Do a round of remeshing
    mesh = remesh.default_trimesh_remesh(mesh, size=2)

    # Orient if needed
    orientation = np.array([0, np.pi / 2 * 0.01, 0])
    # orientation = np.array([0, np.pi/2, 0])
    # orientation = np.array([0, 0.0, 0])

    mesh = trimesh_util.get_transformed_mesh_trs(mesh, orientation=orientation)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    device = torch.device("cuda")

    tensor_verts = torch.from_numpy(mesh_aux.vertices).float().to(device)
    tensor_faces = torch.from_numpy(mesh_aux.faces).float().to(device)
    meshes = Meshes(faces=[tensor_faces], verts=[tensor_verts])

    deform_verts = torch.full(meshes.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = optim.Adam([deform_verts], lr=1e-3)
    # optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

    thickness_predictor = DgcnnThickness()
    thickness_loss_calculator = ThicknessLoss(x_warn=0.1, x_fail=0.05, crossover=0.05)

    plot_every_num = 4

    for i in range(20):
        print("====== iter", i, "===========")
        # Initialize optimizer
        optimizer.zero_grad()

        # Deform the mesh
        new_mesh = meshes.offset_verts(deform_verts)

        if torch.isnan(new_mesh.verts_packed()).any():
            print("nan")

        # smoothing_loss = mesh_laplacian_smoothing(new_mesh, method="uniform") #1e-2
        loss_normal = mesh_normal_consistency(new_mesh) # 1e-1
        overhang_loss = normals.loss_mesh_overhangs(new_mesh) # 1e-4
        # stairstep_loss = normals.loss_mesh_stairsteps(new_mesh) # 1e-4
        # regularization_loss = torch.mean(torch.norm(deform_verts, dim=1))
        thicknesses = thickness_predictor.get_thickness(new_mesh)
        thickness_loss = thickness_loss_calculator.get_loss(thicknesses)

        loss = (
                overhang_loss
                # 1e1 * thickness_loss
                # stairstep_loss
                # + 1e0 * smoothing_loss
                + 1e1 * loss_normal #+
                # + 1e1 * regularization_loss
        )
        # if math.isnan(loss.item()):
        #     print("nan")
        # print("overhang", overhang_loss.item(), "stair", stairstep_loss.item(),
        #       "\nnormal", loss_normal.item(), "smooth", smoothing_loss.item(),
        #       "\nregularization", regularization_loss.item())
        try:
            loss.backward()
        except:
            break
        if torch.isnan(deform_verts.grad).any():
            print("grad nan")


        # Print the losses
        print('total_loss = %.6f' % loss)
        torch.nn.utils.clip_grad_norm_(deform_verts, 1)
        # torch.nn.utils.clip_grad_norm_(loss, 1)
        # Optimization step
        optimizer.step()

        if i % plot_every_num == 0:
            show_mesh(new_mesh)

    # Visualize
    optimized_mesh = trimesh.Trimesh(vertices=new_mesh.verts_packed().detach().cpu().numpy(),
                           faces=new_mesh.faces_packed().detach().cpu().numpy())
    trimesh_util.show_mesh_with_z_normal(mesh)
    trimesh_util.show_mesh_with_z_normal(optimized_mesh)

