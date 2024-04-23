import json

import paths
from dataset.process_and_save import MeshDatasetFileManager
from heuristic_prediction.pointcloud_dataloader import PointCloudDataset
import trimesh_util
from dgcnn_model import DGCNN_segment
import torch
import torch.nn as nn
import trimesh

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


def show_inference_pointcloud(model, cloud):
    cloud_tens = torch.from_numpy(cloud)
    cloud_tens = cloud_tens.to(device)
    preds = model(cloud_tens[None, :, :])
    preds = preds.detach().cpu().numpy()
    preds = preds[:, :, 0].flatten()
    trimesh_util.show_sampled_values(mesh=None, points=cloud, values=preds, normalize=True)


def show_inference_mesh(model, mesh, count):
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    cloud, _ = mesh_aux.sample_and_get_normals(count=count, use_weight="even", return_face_ids=False)
    cloud_tens = torch.from_numpy(cloud).float()
    cloud_tens = cloud_tens.to(device)
    preds = model(cloud_tens[None, :, :])
    preds = preds.detach().cpu().numpy()
    preds = preds[:, :, 0].flatten()
    trimesh_util.show_sampled_values(mesh=mesh, points=cloud, values=preds, normalize=True)

def display_clouds(model, path):
    dataset = PointCloudDataset(path, args['num_points'], label_names=label_names,
                      partition='train',
                      data_fraction=args["data_fraction"], use_numpy=True,
                      sampling_method=args["sampling_method"],
                      outputs_at=args["outputs_at"],
                      imbalance_weight_num_bins=args["imbalanced_weighting_bins"],
                      remove_outlier_ratio=args["remove_outlier_ratio"])
    for i in range(len(dataset)):
        # a = dataset[i]
        cloud, labels, _ = dataset[i]
        show_inference_pointcloud(model, cloud)

def display_meshes(model, path):
    file_manager = MeshDatasetFileManager(path)
    mesh_files = file_manager.get_mesh_files(absolute=True)
    for file in mesh_files:
        mesh = trimesh.load(file)
        show_inference_mesh(model, mesh, args['num_points'])

if __name__=="__main__":
    path = paths.DATA_PATH + "data_primitives/"

    save_path = paths.select_file(choose_type="folder")
    arg_path = save_path + "args.json"
    checkpoint_path = save_path + "model.t7"
    with open(arg_path, 'r') as f:
        model_args = json.load(f)
    model = DGCNN_segment(model_args)
    model = nn.DataParallel(model)
    model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    display_meshes(model, path)








