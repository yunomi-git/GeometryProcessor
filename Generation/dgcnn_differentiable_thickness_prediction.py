# moves from given input mesh to prediction of thicknesses at centroids
from pytorch3d.structures.meshes import Meshes

import json

import paths
from dataset.process_and_save import MeshDatasetFileManager
from shape_regression.pointcloud_dataloader import PointCloudDataset
import trimesh_util
from shape_regression.dgcnn_model import DGCNN_segment
import printability_heuristics.DifferentiableNormals as difnorm
import torch
import torch.nn as nn
import trimesh
from printability_heuristics.Thresholds import get_threshold_penalty
import pyvista as pv
from shape_regression import regression_tools

args = {
    # Dataset Param
    "num_points": 4096,
    "data_fraction": 0.3,
    "sampling_method": "even",
    "imbalanced_weighting_bins": 1, #1 means no weighting
    "remove_outlier_ratio": 0.1, # 0 means remove no outliers
    "outputs_at": "vertices"
}

label_names = ["thickness"]

device = "cuda"

class DgcnnThickness:
    def __init__(self):
        device = "cuda"

        save_path = paths.MODELS_PATH + "th5k_fx/DGCNN_segment/427_19_16_thick-/"
        # save_path = paths.MODELS_PATH + "mcb_scal_a/DGCNN_segment/430_14_0_thick-/"
        arg_path = save_path + "args.json"
        checkpoint_path = save_path + "model.t7"
        model_args = regression_tools.load_args(arg_path)
        self.model = DGCNN_segment(model_args)
        self.modelmodel = nn.DataParallel(self.model)
        self.model.to(device)
        self.model.eval()

        checkpoint = torch.load(checkpoint_path)
        self.modelmodel.load_state_dict(checkpoint)

    def get_thickness(self, meshes: Meshes):
        # first get normals per face
        normals, _ = difnorm.calculate_normals_differentiable(meshes)
        # then get centroids per face
        point_cloud = difnorm.calculate_centroids_differentiable(meshes)
        # concatenate normals and centroids
        augmented_cloud = torch.cat((point_cloud, normals), dim=1)
        # feed to model
        preds = self.model(augmented_cloud[None, :, :])

        return preds

def show_inference_pointcloud(preds, cloud):
    cloud = cloud.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    preds = preds[:, :, 0].flatten()
    trimesh_util.show_sampled_values(mesh=None, points=cloud, values=preds, normalize=True, alpha=1.0)

class ThicknessLoss:
    def __init__(self, x_warn=0.1, x_fail=0.05, crossover=0.05):
        self.loss_func = get_threshold_penalty(x_warn=0.1, x_fail=0.05, crossover=0.05)

    def get_loss(self, thicknesses):
        thickness_loss = self.loss_func(thicknesses)
        loss = torch.mean(thickness_loss)
        # loss = torch.mean(thicknesses)
        return loss

# def get_thickness_loss(thicknesses):
#     loss_func = get_threshold_penalty(x_warn=0.1, x_fail=0.05, crossover=0.05)
#     thickness_loss = loss_func(thicknesses)
#     loss = torch.mean(thickness_loss)
#     return loss

if __name__=="__main__":
    mesh = trimesh.load(paths.HOME_PATH + "stls/squirrel.stl")
    mesh = trimesh_util.normalize_mesh(mesh, center=True, normalize_scale=True)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    tensor_verts = torch.from_numpy(mesh_aux.vertices).float().to(device)
    tensor_faces = torch.from_numpy(mesh_aux.faces).float().to(device)
    meshes = Meshes(faces=[tensor_faces], verts=[tensor_verts])
    meshes.to(device)

    thickness_predictor = DgcnnThickness()
    thicknesses = thickness_predictor.get_thickness(meshes)
    cloud = difnorm.calculate_centroids_differentiable(meshes)

    # plot points and thicknesses
    show_inference_pointcloud(thicknesses, cloud)