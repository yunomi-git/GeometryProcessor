import pyvista_util
from printability_heuristics import DifferentiableNormals as normals
from printability_heuristics.DifferentiableNormals import StairstepLoss, OverhangLoss
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
torch.autograd.set_detect_anomaly(True)


def show_mesh(meshes: Meshes):
    optimized_mesh = trimesh.Trimesh(vertices=meshes.verts_packed().detach().cpu().numpy(),
                           faces=meshes.faces_packed().detach().cpu().numpy())
    # trimesh_util.show_mesh_with_z_normal(optimized_mesh)
    pyvista_util.show_mesh_z_mag(optimized_mesh)

class LossAnalytics:
    def __init__(self):
        self.losses = {}
        self.loss = 0

    def reset(self):
        self.losses = {}
        self.loss = 0

    def set(self, name, loss):
        self.losses[name] = loss.item()
        self.loss += loss

    def print(self):
        for key in list(self.losses.keys()):
            print(key, "\t", self.losses[key])

if __name__ == "__main__":
    # mesh = trimesh.load(paths.DATASETS_PATH + "Dataset_Thingiverse_10k_Remesh2/32770.stl")
    mesh = trimesh.load(paths.HOME_PATH + "stls/thickness_test_pin.stl")
    # Normalize
    mesh = trimesh_util.normalize_mesh(mesh, center=True, normalize_scale=True)
    # Do a round of remeshing
    mesh = remesh.default_trimesh_remesh(mesh, size=2)

    # Orient if needed
    # orientation = np.array([0, np.pi / 2 * 0.01, 0])
    orientation = np.array([0, np.pi/2, 0])
    # orientation = np.array([0, 0.0, 0])

    mesh = trimesh_util.get_transformed_mesh_trs(mesh, orientation=orientation)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    device = torch.device("cuda")

    tensor_verts = torch.from_numpy(mesh_aux.vertices).float().to(device)
    tensor_faces = torch.from_numpy(mesh_aux.faces).float().to(device)
    meshes = Meshes(faces=[tensor_faces], verts=[tensor_verts])

    deform_verts = torch.full(meshes.verts_packed().shape, 0.0, device=device, requires_grad=True)
    # optimizer = optim.Adam([deform_verts], lr=1e-3)
    optimizer = torch.optim.SGD([deform_verts], lr=1e-3, momentum=0.9)

    thickness_predictor = DgcnnThickness()
    thickness_loss_calculator = ThicknessLoss(x_warn=0.2, x_fail=0.1, crossover=0.05)

    stairstep_loss_calculator = StairstepLoss()

    overhang_loss_calculator = OverhangLoss()

    num_iterations = 100
    num_plots = 5
    plot_every_num = num_iterations // num_plots
    loss_analytics = LossAnalytics()

    for i in range(num_iterations):
        loss_analytics.reset()

        # Initialize optimizer
        optimizer.zero_grad()

        # Deform the mesh
        new_mesh = meshes.offset_verts(deform_verts)

        smoothing_loss = mesh_laplacian_smoothing(new_mesh, method="uniform") #1e-2
        loss_normal = mesh_normal_consistency(new_mesh) # 1e-1
        overhang_loss = overhang_loss_calculator.get_loss(new_mesh)
        stairstep_loss = stairstep_loss_calculator.get_loss(new_mesh) # 1e-4
        regularization_loss = torch.mean(torch.norm(deform_verts, dim=1))
        thicknesses = thickness_predictor.get_thickness(new_mesh)
        thickness_loss = thickness_loss_calculator.get_loss(thicknesses)

        # loss_analytics.set("thickness", 1e5 * thickness_loss)
        loss_analytics.set("overhang", overhang_loss * 1e1)
        # loss_analytics.set("stair", stairstep_loss)
        # loss_analytics.set("smooth", 1e0 * smoothing_loss)
        # loss_analytics.set("normal", 1e1 * loss_normal)
        loss_analytics.set("regularization", 1e1 * regularization_loss)

        loss = loss_analytics.loss
        loss.backward()

        # Optimization step
        optimizer.step()

        if i % plot_every_num == 0:
            print("====== iter", i, "===========")
            # Print the losses
            loss_analytics.print()
            print('total_loss = %.6f' % loss)
            torch.nn.utils.clip_grad_norm_(deform_verts, 1)
            show_mesh(new_mesh)

    # Visualize
    optimized_mesh = trimesh.Trimesh(vertices=new_mesh.verts_packed().detach().cpu().numpy(),
                           faces=new_mesh.faces_packed().detach().cpu().numpy())
    # trimesh_util.show_mesh_with_z_normal(mesh)
    # trimesh_util.show_mesh_with_z_normal(optimized_mesh)
    pyvista_util.show_meshes_in_grid([mesh, optimized_mesh], c=2, r=1)


