import json

import paths
from dataset.process_and_save import MeshDatasetFileManager
from shape_regression.diffusionnet_model import DiffusionNetWrapper, DiffusionNetDataset
from shape_regression.pointcloud_dataloader import PointCloudDataset
import trimesh_util
from dgcnn_model import DGCNN_segment
import torch
import torch.nn as nn
import trimesh
import pyvista as pv
import numpy as np
import pyvista_util
from shape_regression import regression_tools

args = {
    # Dataset Param
    "num_points": 2048,
    "data_fraction": 0.01,
    "remove_outlier_ratio": 0.0, # 0 means remove no outliers
    "outputs_at": "vertices"
}

# label_names = ["Thickness"]

device = "cuda"



def plot_mesh_error(vertices, faces, preds, actual, save_path=None, show_deviation=True):
    min_value = min([np.min(preds), np.min(actual)])
    max_value = max([np.max(preds), np.max(actual)])
    mesh = pyvista_util.convert_to_pv_mesh(vertices, faces)

    default_size = 600
    if show_deviation:
        length = 3 * default_size
    else:
        length = 2 * default_size
    hide_gui = False
    if save_path is not None:
        hide_gui = True
    pl = pv.Plotter(shape=(1, 3), window_size=[length, default_size], off_screen=hide_gui)
    pl.subplot(0, 0)
    actor1 = pl.add_mesh(
        mesh,
        scalars=actual.flatten(),
        show_scalar_bar=True,
        scalar_bar_args={'title': 'Actual',
                         'n_labels': 3},
        clim=[min_value, max_value]
    )
    pl.add_text('Actual', color='black')
    actor1.mapper.lookup_table.cmap = 'jet'

    pl.subplot(0, 1)
    actor2 = pl.add_mesh(
        mesh,
        scalars=preds.flatten(),
        show_scalar_bar=True,
        scalar_bar_args={'title': 'Preds',
                         'n_labels': 3},
        clim=[min_value, max_value]

    )
    pl.add_text('Pred', color='black')
    actor2.mapper.lookup_table.cmap = 'jet'

    if show_deviation:
        error = np.abs(preds - actual)
        pl.subplot(0, 2)
        actor3 = pl.add_mesh(
            mesh,
            scalars=error,
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Error',
                             'n_labels': 3},
            clim=[0, np.max(error)]

        )
        pl.add_text('Error', color='black')
        actor3.mapper.lookup_table.cmap = 'Reds'

    pl.link_views()
    if save_path is None:
        pl.show()
    else:
        pl.screenshot(save_path)

def plot_cloud_error(cloud, preds, actual, save_path=None, show_deviation=True):
    min_value = min([np.min(preds), np.min(actual)])
    min_value = min(0, min_value)
    max_value = max([np.max(preds), np.max(actual)])

    default_size = 600
    if show_deviation:
        length = 3 * default_size
    else:
        length = 2 * default_size

    hide_gui = False
    if save_path is not None:
        hide_gui = True
    pl = pv.Plotter(shape=(1, 3), window_size=[length, default_size], off_screen=hide_gui)
    pl.subplot(0, 0)
    actor1 = pl.add_points(
        cloud,
        scalars=actual.flatten(),
        render_points_as_spheres=True,
        point_size=10,
        show_scalar_bar=True,
        scalar_bar_args={'title': 'Actual',
                         'n_labels': 3},
        clim=[min_value, max_value]
    )
    pl.add_text('Actual', color='black')
    actor1.mapper.lookup_table.cmap = 'jet'

    pl.subplot(0, 1)
    actor2 = pl.add_points(
        cloud,
        scalars=preds,
        render_points_as_spheres=True,
        point_size=10,
        show_scalar_bar=True,
        scalar_bar_args={'title': 'Predictions',
                         'n_labels': 3},
        clim=[min_value, max_value]

    )
    pl.add_text('Pred', color='black')
    actor2.mapper.lookup_table.cmap = 'jet'

    if show_deviation:
        error = np.abs(preds - actual)
        pl.subplot(0, 2)
        actor3 = pl.add_points(
            cloud,
            scalars=error,
            render_points_as_spheres=True,
            point_size=10,
            scalar_bar_args={'title': 'Error',
                             'n_labels': 3},
            clim=[0, np.max(error)]
        )
        pl.add_text('Error', color='black')
        actor3.mapper.lookup_table.cmap = 'Reds'

    pl.link_views()
    if save_path is None:
        pl.show()
    else:
        pl.screenshot(save_path)

def display_meshes(model, dataset, categorical):
    for i in range(len(dataset)):
        verts, faces, label = dataset[i]

        # cloud_tens = torch.from_numpy(augmented_cloud).float()
        verts = verts.to(device)
        faces = faces.to(device)
        preds = model(verts[None, :, :], faces[None, :, :])
        preds = preds.detach().cpu().numpy()

        if not categorical:
            preds = preds[:, :, 0].flatten()
        else:
            preds = np.argmax(preds, axis=-1).flatten()
        plot_cloud_error(cloud=verts.detach().cpu().numpy(),
                        preds=preds,
                        actual=label.detach().cpu().numpy().flatten())
        # plot_mesh_error(vertices=verts.detach().cpu().numpy(),
        #                 faces=faces.detach().cpu().numpy(),
        #                 preds=preds,
        #                 actual=label.detach().cpu().numpy().flatten())


def display_point_clouds(model, dataset, categorical):
    for i in range(len(dataset)):
        augmented_cloud, actual = dataset[i]
        cloud = augmented_cloud[:, :3]

        cloud_tens = torch.from_numpy(augmented_cloud).float()
        cloud_tens = cloud_tens.to(device)
        preds = model(cloud_tens[None, :, :])
        preds = preds.detach().cpu().numpy()

        if not categorical:
            preds = preds[:, :, 0].flatten()
        else:
            preds = np.argmax(preds, axis=-1).flatten()
        plot_cloud_error(cloud, preds, actual.flatten())

if __name__=="__main__":
    path = paths.CACHED_DATASETS_PATH + "DaVinci/train/"
    op_cache_dir = path + "../op_cache/"

    mesh_visualization = True

    save_path = paths.select_file(choose_type="folder")
    arg_path = save_path + "args.json"
    checkpoint_path = save_path + "model.t7"

    model_args = regression_tools.load_args(arg_path)

    if mesh_visualization:
        model = DiffusionNetWrapper(model_args, op_cache_dir=op_cache_dir, device=device)
    else:
        model = DGCNN_segment(model_args)

    if "data_parallel" not in model_args or model_args["data_parallel"]:
        model = nn.DataParallel(model.to(device))    # model = nn.DataParallel(model)
    model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    categorical=True
    if model_args["use_category_thresholds"] is None:
        categorical=False

    if mesh_visualization:
        dataset = DiffusionNetDataset(path, model_args["k_eig"],
                                      args=model_args,
                                      op_cache_dir=op_cache_dir,
                                      partition="train",
                                      num_data=None,
                                      data_fraction=0.01,
                                      augment_random_rotate=False,
                                      use_imbalanced_weights=True,
                                      augmentations="none",
                                      remove_outlier_ratio=0.0,
                                      cache_operators=False,
                                      aggregator=None)
        display_meshes(model, dataset, categorical=categorical)
    else:
        dataset = PointCloudDataset(path, 4096, args=model_args,
                                    partition='test',
                                    data_fraction=0.1,
                                    augmentations=None,
                                    remove_outlier_ratio=args["remove_outlier_ratio"])

        display_point_clouds(model, dataset, categorical=categorical)





