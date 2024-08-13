# GOals:
# 1. Plot default + augmentations
# 2. Plot multiple defaults
# Plot with vertex labels
# Plot the meshes

import dataset.process_and_save_temp as pas2
import paths
from dataset.process_and_save import MeshDatasetFileManager
# from shape_regression.pointcloud_dataloader import PointCloudDataset
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
    # print(len(mesh_labels))
    # print(nc)
    pl = pv.Plotter(shape=(nr, nc))

    for i in range(len(mesh_labels)):
        mesh_label = mesh_labels[i]
        mesh = convert_to_pv_mesh(mesh_label.vertices, mesh_label.faces)
        labels = mesh_label.get_vertex_labels(vertex_label_name)
        # print(i // nc, i % nc)
        pl.subplot(i // nc, i % nc)
        pl.add_mesh(mesh, show_edges=show_edge, scalars=labels, show_scalar_bar=True)
        # actor.mapper.lookup_table.cmap = 'jet'
        pl.show_bounds(grid=True, all_edges=False,  font_size=10)

    pl.link_views()
    pl.show()



def plot_all_clouds(c, r, vertices, labels=None):
    num_data = len(vertices)
    num_visualized = c * r
    num_iterations = int(np.ceil(num_data / num_visualized))

    for i in range(num_iterations):
        pl = pv.Plotter(shape=(r, c))
        for ri in range(r):
            for ci in range(c):
                cloud = vertices[i * r * c + ri * c + ci]
                cloud_labels = labels[i * r * c + ri * c + ci]

                pl.subplot(ri, ci)
                actor = pl.add_points(
                    points=cloud,
                    scalars=cloud_labels,
                    render_points_as_spheres=True,
                    point_size=5,
                    show_scalar_bar=True,
                    # text="Curvature"
                )
                # actor.mapper.lookup_table.cmap = 'jet'
                pl.show_bounds(grid=True, all_edges=False,  font_size=10)

        pl.link_views()
        pl.show()

def plot_all_meshes(c, r, vertices, faces, labels):
    num_data = len(vertices)
    num_visualized = c * r
    num_iterations = int(np.ceil(num_data / num_visualized))

    for i in range(num_iterations):
        pl = pv.Plotter(shape=(r, c))
        for ri in range(r):
            for ci in range(c):
                idx = i * r * c + ri * c + ci
                if idx >= num_data:
                    break
                mesh_vertices = vertices[idx]
                mesh_faces = faces[idx]
                mesh_labels = labels[idx]
                mesh = convert_to_pv_mesh(mesh_vertices, mesh_faces)
                pl.subplot(ri, ci)
                actor = pl.add_mesh(
                    mesh,
                    scalars=mesh_labels,
                    # render_points_as_spheres=True,
                    # point_size=5,
                    # rgb=True,
                    show_scalar_bar=True,
                    # text="Curvature"
                )
                # actor.mapper.lookup_table.cmap = 'jet'
                pl.show_bounds(grid=True, all_edges=False,  font_size=10)

        pl.link_views()
        pl.show()

if __name__=="__main__":
    path = paths.CACHED_DATASETS_PATH + "DrivAerNet/train/"
    dataset_manager = pas2.DatasetManager(path)
    num_values = 100


    mesh_folders = dataset_manager.get_mesh_folders(10)
    label_names = ["Thickness"]

    # This specifically plots the augmentations
    # for mesh_folder in mesh_folders:
    #     plot_augmentations(mesh_folder, label_names, show_edge=False)

    show_points = True

    if show_points:
    # get point clouds
        vertices, _, labels = dataset_manager.load_numpy_pointcloud(num_clouds=num_values, num_points=1024,
                                                                  augmentations="none", outputs_at = "vertices",
                                                                  desired_label_names=label_names)

        print("avg_position: ", np.mean(np.mean(vertices, axis=1), axis=0))
        print("min_position: ", np.min(np.min(vertices, axis=1), axis=0))
        print("max_position: ", np.max(np.max(vertices, axis=1), axis=0))
        lengths = np.max(vertices, axis=1) - np.min(vertices, axis=1)

        print("avg_length: ", np.mean(lengths, axis=0))
        print("min_length: ", np.min(lengths, axis=0))
        print("max_length: ", np.max(lengths, axis=0))

        plot_all_clouds(6, 6, vertices=vertices, labels=labels)


    # get meshes
    else:
        vertices, _, faces, labels = dataset_manager.load_numpy_meshes(num_meshes=num_values,
                                                                       augmentations="none",
                                                                       outputs_at="vertices",
                                                                       desired_label_names=label_names)
        plot_all_meshes(6, 6, vertices=vertices, faces=faces, labels=labels)
    # for i in range(len(dataset)):
    #     # a = dataset[i]
    #     cloud, labels, _ = dataset[i]
    #     labels = labels[:, 0]
    #     trimesh_util.show_sampled_values(mesh=None, points=cloud, values=labels, normalize=True)


