import numpy as np
import torch
from tqdm import tqdm
from run_visualize_outputs import plot_cloud_error
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
from shape_regression import regression_tools
from shape_regression.run_visualize_outputs import plot_mesh_error


# gather a list of items to test (pc or mesh form)
# for each individually, calculate the loss
# Need: inputs <-> Losses
# find the top n and bottom n in loss
# grab pictures of the pc or meshes

def extract_checkpoint(full_checkpoint):
    model_checkpoint = {}
    for key in list(full_checkpoint.keys()):
        new_key = key[key.find(".") + 1:]
        model_checkpoint[new_key] = full_checkpoint[key]
    return model_checkpoint

def get_evaluations_pointcloud(model, regression_tool, data_loader, device, n):
    model.eval()

    losses = []
    pc_list = []
    train_pred = []
    train_true = []
    # loss_criterion = torch.nn.MSELoss()

    for (data_batch, label_batch) in tqdm(data_loader):
        data_batch, label_batch = data_batch.to(device), label_batch.to(device)
        pred_batch = model(data_batch)

        # look at each item in this batch
        for i in range(len(data_batch)):
            label = label_batch[i]
            pred = pred_batch[i]
            loss = regression_tool.get_loss(preds=pred, label=label)

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

def get_evaluations_mesh(model, regression_tool, data_loader, device, n):
    model.eval()

    losses = []
    vertices_list = []
    faces_list = []
    train_pred = []
    train_true = []

    for (vertices_batch, faces_batch, label_batch) in tqdm(data_loader):
        vertices_batch, faces_batch, label_batch = vertices_batch.to(device), faces_batch.to(device), label_batch.to(device)
        pred_batch = model(vertices_batch, faces_batch)

        # look at each item in this batch
        for i in range(len(vertices_batch)):
            label = label_batch[i]
            pred = pred_batch[i]
            loss = regression_tool.get_loss(preds=pred, label=label)

            losses.append(loss.detach().cpu().numpy())
            vertices_list.append(vertices_batch[i].detach().cpu().numpy())
            faces_list.append(faces_batch[i].detach().cpu().numpy())
            train_true.append(label.cpu().numpy())
            train_pred.append(pred.detach().cpu().numpy())

    losses = np.array(losses)
    sorted_indices = np.argsort(losses) # ascending losses

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

    # if plotting_mesh:
    #     for i in range(len(top_n)):
    #         top_output = top_n[i]
    #         save_path = save_directory + "top_" + str(i) + ".png"
    #         plot_mesh_error(vertices=top_output["vertices"],
    #                         faces=top_output["faces"],
    #                          preds=top_output["preds"],
    #                          actual=top_output["labels"],
    #                          save_path=save_path)
    #         worst_output = worst_n[i]
    #         save_path = save_directory + "worst_" + str(i) + ".png"
    #         plot_mesh_error(vertices=worst_output["vertices"],
    #                         faces=worst_output["faces"],
    #                          preds=worst_output["preds"],
    #                          actual=worst_output["labels"],
    #                          save_path=save_path)
    #
    # else:
    for i in range(len(top_n)):
        top_output = top_n[i]
        save_path = save_directory + "top_" + str(i) + ".png"
        plot_cloud_error(cloud=top_output["vertices"],
                         preds=top_output["preds"],
                         actual=top_output["labels"],
                         save_path=save_path)
        worst_output = worst_n[i]
        save_path = save_directory + "worst_" + str(i) + ".png"
        plot_cloud_error(cloud=worst_output["vertices"],
                         preds=worst_output["preds"],
                         actual=worst_output["labels"],
                         save_path=save_path)


def plot_and_save_global(top_n, worst_n):
    # only save pictures and their numbers
    pass


label_names = ["Thickness"]

device = "cuda"

if __name__=="__main__":

    ###### Run parameters
    dataset_name = "DaVinci/train/"
    use_mesh = False
    #######

    regression_tools.seed_all(0)
    path = paths.CACHED_DATASETS_PATH + dataset_name


    load_path = paths.select_file(choose_type="folder")
    arg_path = load_path + "args.json"
    checkpoint_path = load_path + "model.t7"
    model_args = regression_tools.load_args(arg_path)

    if use_mesh:
        op_cache_dir = path + "../op_cache/"
        model = DiffusionNetWrapper(model_args, op_cache_dir=op_cache_dir, device=device)
    else:
        model = DGCNN_segment(model_args, device=device)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not use_mesh:
        checkpoint = extract_checkpoint(checkpoint)
    model.load_state_dict(checkpoint)

    if use_mesh:
        dataset = DiffusionNetDataset(path, model_args["k_eig"], args=model_args,
                                      partition="test",
                                      data_fraction=0.1,
                                      augment_random_rotate=model_args["augment_random_rotate"],
                                      use_imbalanced_weights=False,
                                      augmentations="none",
                                      cache_operators=False,
                                      aggregator=None)
    else:
        dataset = PointCloudDataset(path, 4096,
                                    args=model_args,
                                    partition='train',
                                    data_fraction=0.1,
                                    use_imbalanced_weights=False)

    dataloader = DataLoader(dataset,
                            num_workers=24,
                            batch_size=1, shuffle=True, drop_last=True)

    regression_tool = regression_tools.RegressionTools(model_args, label_names, train_loader=None,
                                                       test_loader=dataloader, model=model,
                                                       opt=None, scheduler=None, clip_parameters=False,
                                                       use_mesh=use_mesh)

    print("#"*5 + "Evaluationg")
    if use_mesh:
        top_n, worst_n = get_evaluations_mesh(model, regression_tool=regression_tool, data_loader=dataloader, device=device, n=10)
    else:
        top_n, worst_n = get_evaluations_pointcloud(model, regression_tool=regression_tool, data_loader=dataloader, device=device, n=10)

    plot_and_save_vertices(top_n, worst_n, save_directory=load_path + "vis_top_worst_" + dataset_name + "/")
