import numpy as np
import torch
from tqdm import tqdm
from dgcnn_visualize_outputs import plot_cloud_error
import json
from torch.utils.data import DataLoader
from pathlib import Path
import paths
from dataset.process_and_save import MeshDatasetFileManager
from shape_regression.pointcloud_dataloader import PointCloudDataset
import trimesh_util
from shape_regression.diffusionnet_model import DiffusionNetDataset, DiffusionNetWrapper
from dgcnn_model import DGCNN_segment
import torch
import numpy as np
# class EvaluationOutput:
#     def __init__(self, vertices, preds, actuals, faces=None):
#         self.vertices = vertices
#         self.faces = faces
#         self.preds = preds
#         self.actuals = actuals

def extract_checkpoint(full_checkpoint):
    model_checkpoint = {}
    for key in list(full_checkpoint.keys()):
        new_key = key[key.find(".") + 1:]
        model_checkpoint[new_key] = full_checkpoint[key]
    return model_checkpoint

def get_evaluations_pointcloud(model, data_loader, device, n, use_weighting=False):
    model.eval()

    losses = []
    pc_list = []
    train_pred = []
    train_true = []
    loss_criterion = torch.nn.MSELoss()

    for batch_idx, (data_batch, label_batch) in enumerate(tqdm(data_loader)):
        # calculate weights if appropriate
        weight = 1
        if use_weighting and data_loader.dataset.imbalanced_weighting is not None:
            weight = data_loader.dataset.imbalanced_weighting.get_weights(label_batch.numpy())

        data_batch, label_batch = data_batch.to(device), label_batch.to(device)
        pred_batch = model(data_batch)

        # look at each item in this batch
        for i in range(len(data_batch)):
            label = label_batch[i]
            pred = pred_batch[i]
            if use_weighting and data_loader.dataset.imbalanced_weighting is not None:
                weight = data_loader.dataset.imbalanced_weighting.get_weights(label.numpy())
                weight = torch.Tensor(weight)
                weight = weight.to(device)
            loss = loss_criterion(weight * pred, weight * label)

            losses.append(loss.detach().cpu().numpy())
            data = data_batch[i].detach().cpu().numpy()
            pc_list.append(data[:, :3])
            train_true.append(label.cpu().numpy())
            train_pred.append(pred.detach().cpu().numpy())

    losses = np.array(losses)
    sorted_indices = np.argsort(losses) # ascending order

    # grab the top n
    top_n_ind = sorted_indices[:n]
    worst_n_ind = sorted_indices[-n:]
    top_n = []
    worst_n = []
    for i in range(n):
        top_n.append({
            "vertices": pc_list[top_n_ind[i]],
            "labels": train_true[top_n_ind[i]],
            "preds": train_pred[top_n_ind[i]]
        })
        worst_n.append({
            "vertices": pc_list[worst_n_ind[i]],
            "labels": train_true[worst_n_ind[i]],
            "preds": train_pred[worst_n_ind[i]]
        })
    return top_n, worst_n

def get_evaluations_mesh(model, data_loader, device, n, use_weighting=False):
    model.eval()

    losses = []
    vertices_list = []
    faces_list = []
    train_pred = []
    train_true = []
    loss_criterion = torch.nn.MSELoss()

    for batch_idx, (vertices_batch, faces_batch, label_batch) in enumerate(tqdm(data_loader)):
        # calculate weights if appropriate
        weight = 1
        if use_weighting and data_loader.dataset.imbalanced_weighting is not None:
            weight = data_loader.dataset.imbalanced_weighting.get_weights(label_batch.numpy())

        vertices_batch, faces_batch, label_batch = vertices_batch.to(device), faces_batch.to(device), label_batch.to(device)
        pred_batch = model(vertices_batch, faces_batch)

        # look at each item in this batch
        for i in range(len(vertices_batch)):
            label = label_batch[i]
            pred = pred_batch[i]
            if use_weighting and data_loader.dataset.imbalanced_weighting is not None:
                weight = data_loader.dataset.imbalanced_weighting.get_weights(label.numpy())
                weight = torch.Tensor(weight)
                weight = weight.to(device)
            loss = loss_criterion(weight * pred, weight * label)

            losses.append(loss.detach().cpu().numpy())
            vertices_list.append(vertices_batch[i].detach().cpu().numpy())
            faces_list.append(faces_batch[i].detach().cpu().numpy())
            train_true.append(label.cpu().numpy())
            train_pred.append(pred.detach().cpu().numpy())

    losses = np.array(losses)
    sorted_indices = np.argsort(losses) # ascending order

    # grab the top n
    top_n_ind = sorted_indices[:n]
    worst_n_ind = sorted_indices[-n:]
    top_n = []
    worst_n = []
    for i in range(n):
        top_n.append({
            "vertices": vertices_list[top_n_ind[i]],
            "faces": faces_list[top_n_ind[i]],
            "labels": train_true[top_n_ind[i]],
            "preds": train_pred[top_n_ind[i]]
        })
        worst_n.append({
            "vertices": vertices_list[worst_n_ind[i]],
            "faces": faces_list[worst_n_ind[i]],
            "labels": train_true[worst_n_ind[i]],
            "preds": train_pred[worst_n_ind[i]]
        })
    return top_n, worst_n

def plot_and_save_vertices(top_n, worst_n, save_directory):
    plotting_mesh = False
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    if "faces" in top_n[0]:
        plotting_mesh = True

    if plotting_mesh:
        print("not implemented")
    else:
        for i in range(len(top_n)):
            top_output = top_n[i]
            save_path = save_directory + "top_" + str(i) + ".svg"
            plot_cloud_error(cloud=top_output["vertices"],
                             preds=top_output["preds"],
                             actual=top_output["labels"],
                             save_path=save_path)
            worst_output = worst_n[i]
            save_path = save_directory + "worst_" + str(i) + ".svg"
            plot_cloud_error(cloud=worst_output["vertices"],
                             preds=worst_output["preds"],
                             actual=worst_output["labels"],
                             save_path=save_path)


def plot_and_save_global(top_n, worst_n):
    # only save pictures and their numbers
    pass

args = {
    # Dataset Param
    "num_points": 4096,
    "data_fraction": 0.1,
    "remove_outlier_ratio": 0.0, # 0 means remove no outliers
    "outputs_at": "vertices",
    "data_parallel": False,
}

label_names = ["Thickness"]

device = "cuda"

if __name__=="__main__":
    # gather a list of items to test (pc or mesh form)
    # for each individually, calculate the loss
    # Need: inputs <-> Losses
    # find the top n and bottom n in loss
    # grab pictures of the pc or meshes
    dataset_name = "DrivAerNet/train/"
    path = paths.CACHED_DATASETS_PATH + dataset_name

    use_mesh = True

    load_path = paths.select_file(choose_type="folder")
    arg_path = load_path + "args.json"
    checkpoint_path = load_path + "model.t7"
    with open(arg_path, 'r') as f:
        model_args = json.load(f)

    if use_mesh:
        op_cache_dir = dataset_name + "../"
        model = DiffusionNetWrapper(model_args, op_cache_dir=op_cache_dir, device=device)
    else:
        model = DGCNN_segment(model_args, device=device)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not use_mesh:
        checkpoint = extract_checkpoint(checkpoint)
    model.load_state_dict(checkpoint)

    if use_mesh:
        dataset = DiffusionNetDataset(path, model_args["k_eig"],
                                      partition="train",
                                      data_fraction=args["data_fraction"], label_names=label_names,
                                      augment_random_rotate=model_args["augment_random_rotate"],
                                      extra_vertex_label_names=model_args["input_append_vertex_label_names"],
                                      extra_global_label_names=model_args["input_append_global_label_names"],
                                      outputs_at=model_args["outputs_at"],
                                      use_imbalanced_weights=False,
                                      augmentations=model_args["augmentations"],
                                      remove_outlier_ratio=0.0,
                                      cache_operators=False,
                                      aggregator=None),
    else:
        dataset = PointCloudDataset(path, args['num_points'], label_names=label_names,
                                    append_label_names=model_args['input_append_label_names'],
                                    partition='train',
                                    data_fraction=args["data_fraction"],
                                    outputs_at=args["outputs_at"],
                                    use_imbalanced_weights=False,
                                    remove_outlier_ratio=args["remove_outlier_ratio"])

    dataloader = DataLoader(dataset,
                            num_workers=24,
                            batch_size=1, shuffle=True, drop_last=True)
    if use_mesh:
        top_n, worst_n = get_evaluations_mesh(model, data_loader=dataloader, device=device, n=5,
                                                    use_weighting=False)
    else:
        top_n, worst_n = get_evaluations_pointcloud(model, data_loader=dataloader, device=device, n=5,
                                                    use_weighting=False)
    plot_and_save_vertices(top_n, worst_n, save_directory=load_path + "vis_top_worst_" + dataset_name + "/")
