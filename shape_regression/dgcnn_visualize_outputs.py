import json

import paths
from dataset.process_and_save import MeshDatasetFileManager
from shape_regression.pointcloud_dataloader import PointCloudDataset
import trimesh_util
from dgcnn_model import DGCNN_segment
import torch
import torch.nn as nn
import trimesh
import pyvista as pv
import numpy as np
import pyvista_util

args = {
    # Dataset Param
    "num_points": 4096,
    "data_fraction": 0.1,
    "imbalanced_weighting_bins": 1, #1 means no weighting
    "remove_outlier_ratio": 0.0, # 0 means remove no outliers
    "outputs_at": "vertices"
}

label_names = ["Thickness"]

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
    pl = pv.Plotter(shape=(1, 3), window_size=[length, default_size])
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
    actor2 = pl.add_points(
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
        pl.subplot(0, 2)
        actor3 = pl.add_points(
            mesh,
            scalars=np.abs(preds - actual),
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Error',
                             'n_labels': 3},

        )
        pl.add_text('Error', color='black')
        actor3.mapper.lookup_table.cmap = 'Reds'

    pl.link_views()
    if save_path is None:
        pl.show()
    else:
        pl.save_graphic(save_path)

def plot_cloud_error(cloud, preds, actual, save_path=None, show_deviation=True):
    min_value = min([np.min(preds), np.min(actual)])
    max_value = max([np.max(preds), np.max(actual)])

    default_size = 600
    if show_deviation:
        length = 3 * default_size
    else:
        length = 2 * default_size
    pl = pv.Plotter(shape=(1, 3), window_size=[length, default_size])
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
        pl.save_graphic(save_path)

def comparison(model, dataloader, i, categorical):
    augmented_cloud, actual = dataloader[i]
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




def display_meshes(model, dataset, categorical):
    for i in range(len(dataset)):
        comparison(model, dataset, i, categorical=categorical)

if __name__=="__main__":
    path = paths.CACHED_DATASETS_PATH + "DaVinci/train/"

    save_path = paths.select_file(choose_type="folder")
    arg_path = save_path + "args.json"
    checkpoint_path = save_path + "model.t7"
    with open(arg_path, 'r') as f:
        model_args = json.load(f)
    model = DGCNN_segment(model_args)
    # model = nn.DataParallel(model)
    model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    use_category_thresholds = None
    if "use_category_thresholds" in model_args:
        use_category_thresholds = model_args["use_category_thresholds"]
    categorical=True
    if use_category_thresholds is None:
        categorical=False

    dataset = PointCloudDataset(path, args['num_points'], label_names=label_names,
                                append_label_names=model_args['input_append_label_names'],
                                partition='train',
                                data_fraction=0.1,
                                outputs_at=args["outputs_at"],
                                augmentations=None,
                                categorical_thresholds=use_category_thresholds,
                                # imbalance_weight_num_bins=args["imbalanced_weighting_bins"],
                                remove_outlier_ratio=args["remove_outlier_ratio"])

    display_meshes(model, dataset, categorical=categorical)





# def show_inference_pointcloud(model, cloud):
#     cloud_tens = torch.from_numpy(cloud)
#     cloud_tens = cloud_tens.to(device)
#     preds = model(cloud_tens[None, :, :])
#     preds = preds.detach().cpu().numpy()
#     preds = preds[:, :, 0].flatten()
#     trimesh_util.show_sampled_values(mesh=None, points=cloud, values=preds, normalize=True)
#
#
# def show_inference_mesh(model, mesh, count):
#     mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
#     cloud, actual = mesh_aux.calculate_thicknesses_samples(count=count)
#     # cloud, normals = mesh_aux.sample_and_get_normals(count=count, use_weight="even", return_face_ids=False)
#     cloud_tens = torch.from_numpy(cloud).float()
#     cloud_tens = cloud_tens.to(device)
#     preds = model(cloud_tens[None, :, :])
#     preds = preds.detach().cpu().numpy()
#     preds = preds[:, :, 0].flatten()
#
#     # hit_origins, wall_thicknesses = mesh_aux.calculate_thickness_at_points(cloud, normals)
#     error = actual
#     trimesh_util.show_sampled_values(mesh=mesh, points=cloud, values=error, scale=[0, 1])
#
#     # trimesh_util.show_sampled_values(mesh=mesh, points=cloud, values=preds, normalize=True)

# def display_clouds(model, path):
#     dataset = PointCloudDataset(path, args['num_points'], label_names=label_names,
#                       partition='train',
#                       data_fraction=args["data_fraction"],
#                       outputs_at=args["outputs_at"],
#                       # imbalance_weight_num_bins=args["imbalanced_weighting_bins"],
#                       remove_outlier_ratio=args["remove_outlier_ratio"])
#     for i in range(len(dataset)):
#         # a = dataset[i]
#         cloud, labels, _ = dataset[i]
#         show_inference_pointcloud(model, cloud)


