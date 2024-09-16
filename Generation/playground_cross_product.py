import torch
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
import torch.nn.functional as f

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
device = torch.device("cuda")

def show_mesh(meshes: Meshes):
    optimized_mesh = trimesh.Trimesh(vertices=meshes.verts_packed().detach().cpu().numpy(),
                           faces=meshes.faces_packed().detach().cpu().numpy())
    # trimesh_util.show_mesh_with_z_normal(optimized_mesh)
    pyvista_util.show_mesh_z_mag(optimized_mesh)

def line_only():

    v1 = torch.Tensor([0.0001, 1, 1]).to(device)
    v1.requires_grad_()
    v2 = torch.Tensor([0, 0, 0.00001]).to(device)
    loss_crit = torch.nn.MSELoss()
    optimizer = optim.Adam([v1], lr=1e-3)

    num_iterations = 1000
    for i in range(num_iterations):
        optimizer.zero_grad()

        n0 = torch.linalg.cross(v1, v2)
        n0 = f.normalize(n0, p=2, dim=0)

        loss = torch.mean(n0) + torch.abs(v1).sum()
        loss.backward()

        optimizer.step()

        print("n0", n0)
        print(loss)
        print(v1)
        print()



def mesh():
    vertices = np.array([[0.0, 0, 0],
                         [1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
    faces = np.array([[0, 1, 2],
                     [1, 2, 3],
                     [2, 3, 0],
                     [0, 1, 3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    trimesh.repair.fix_normals(mesh)
    # orientation = np.array([0, np.pi/2, 0])
    # orientation = np.array([0, 0.0, 0])

    # mesh = trimesh_util.get_transformed_mesh_trs(mesh, orientation=orientation)

    tensor_verts = torch.from_numpy(mesh.vertices).float().to(device)
    tensor_faces = torch.from_numpy(mesh.faces).float().to(device)
    meshes = Meshes(faces=[tensor_faces], verts=[tensor_verts])

    deform_verts = torch.full(meshes.verts_packed().shape, 0.0, device=device, requires_grad=False)
    moving_val = torch.full([1], 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([deform_verts], lr=1e-3, momentum=0.9)

    overhang_loss_calculator = OverhangLoss()

    show_mesh(meshes)
    for i in range(1000):
        optimizer.zero_grad()

        deform_verts[3, 2] = moving_val
        new_mesh = meshes.offset_verts(deform_verts)
        overhang_loss = overhang_loss_calculator.get_loss(new_mesh)
        print(overhang_loss)
        print(moving_val.grad)

        overhang_loss.backward()

        optimizer.step()

        show_mesh(new_mesh)

if __name__ == "__main__":
    mesh()
