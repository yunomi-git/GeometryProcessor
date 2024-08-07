# GOals:
# 1. Plot default + augmentations
# 2. Plot multiple defaults
# Plot with vertex labels
# Plot the meshes

import dataset.process_and_save_temp as pas2
import paths
from dataset.process_and_save import MeshDatasetFileManager
from heuristic_prediction.pointcloud_dataloader import PointCloudDataset
import trimesh_util
import pyvista as pv
import numpy as np
import trimesh

def convert_to_pv_mesh(vertices, faces):
    pad = 3.0 * np.ones((len(faces), 1))
    faces = np.concatenate((pad, faces), axis=1)
    faces = np.hstack(faces).astype(np.int64)
    mesh = pv.PolyData(vertices, faces)
    return mesh


label_names = ["thickness"]

def plot_augmentations(mesh_folder: pas2.MeshFolder, vertex_label_name, show_edge=False):
    mesh_labels = mesh_folder.load_all_augmentations()
    nr = 2
    nc = int(np.ceil(len(mesh_labels) / nr))
    print(len(mesh_labels))
    print(nc)
    pl = pv.Plotter(shape=(nr, nc))

    for i in range(len(mesh_labels)):
        mesh_label = mesh_labels[i]
        mesh = convert_to_pv_mesh(mesh_label.vertices, mesh_label.faces)
        labels = mesh_label.get_vertex_labels(vertex_label_name)
        print(i // nc, i % nc)
        pl.subplot(i // nc, i % nc)
        pl.add_mesh(mesh, show_edges=show_edge, scalars=labels, show_scalar_bar=True)
        # actor.mapper.lookup_table.cmap = 'jet'
        pl.show_bounds(grid=True, all_edges=False,  font_size=10)

    pl.link_views()
    pl.show()



def plot_all(c, r, dataset):
    num_data = len(dataset)
    num_visualized = c * r
    num_iterations = int(np.ceil(num_data / num_visualized))

    for i in range(num_iterations):
        pl = pv.Plotter(shape=(r, c))
        for ri in range(r):
            for ci in range(c):
                cloud, _, _ = dataset[i * r * c + ri * c + ci]
                pl.subplot(ri, ci)
                actor = pl.add_points(
                    points=cloud,
                    # scalars=curvature,
                    render_points_as_spheres=True,
                    point_size=5,
                    show_scalar_bar=True,
                    # text="Curvature"
                )
                # actor.mapper.lookup_table.cmap = 'jet'
                pl.show_bounds(grid=True, all_edges=False,  font_size=10)

        pl.link_views()
        pl.show()

if __name__=="__main__":
    path = paths.CACHED_DATASETS_PATH + "th10k_norm/train/"
    dataset_manager = pas2.DatasetManager(path)
    mesh_folders = dataset_manager.get_mesh_folders(10)
    label_names = ["Thickness"]

    for mesh_folder in mesh_folders:
        plot_augmentations(mesh_folder, label_names, show_edge=False)

    # all_clouds = dataset.point_clouds
    # # data x point x position
    #
    #
    # print("avg_position: ", np.mean(np.mean(dataset.point_clouds, axis=1), axis=0))
    # print("min_position: ", np.min(np.min(dataset.point_clouds, axis=1), axis=0))
    # print("max_position: ", np.max(np.max(dataset.point_clouds, axis=1), axis=0))
    # lengths = np.max(dataset.point_clouds, axis=1) - np.min(dataset.point_clouds, axis=1)
    #
    # print("avg_length: ", np.mean(lengths, axis=0))
    # print("min_length: ", np.min(lengths, axis=0))
    # print("max_length: ", np.max(lengths, axis=0))
    #
    # plot_all(6, 6, dataset)
    # for i in range(len(dataset)):
    #     # a = dataset[i]
    #     cloud, labels, _ = dataset[i]
    #     labels = labels[:, 0]
    #     trimesh_util.show_sampled_values(mesh=None, points=cloud, values=labels, normalize=True)


