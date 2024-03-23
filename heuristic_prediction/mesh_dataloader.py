import numpy as np
import json
import math
import trimesh

import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d

import diffusion_net
from tqdm import tqdm

from dataset.process_and_save import MeshDatasetFileManager, get_augmented_mesh




class DiffusionNetDataset(Dataset):
    def __init__(self, data_root_dir, split_size, k_eig, exclude_dict=None, op_cache_dir=None):
        self.file_manager = MeshDatasetFileManager(data_root_dir)
        self.root_dir = data_root_dir
        self.split_size = split_size  # pass None to take all entries (except those in exclude_dict)
        self.k_eig = k_eig
        self.k_eig_list = []
        self.op_cache_dir = op_cache_dir

        self.entries = {}

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.outputs_list = []

        # Open the base directory and get the contents
        data_files = self.file_manager.get_target_files(absolute=True)

        # Now parse through all the files
        for data_file in tqdm(data_files):
            # print(data_file)
            # Load the file
            file_path = data_file
            with open(file_path, 'r') as f:
                mesh_data = json.load(f)

            if mesh_data["vertices"] > 1e4:
                continue
            if mesh_data["vertices"] < 1e2:
                continue
            if math.isnan(mesh_data["thickness_violation"]):
                continue
            mesh_path = data_root_dir + mesh_data["mesh_relative_path"]

            verts, faces = pp3d.read_mesh(mesh_path)
            verts = torch.tensor(verts).float()
            faces = torch.tensor(faces)

            # Attempt to get eigen decomposition. If cannot, skip
            try:
                diffusion_net.geometry.get_operators(verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
            except: # Or valueerror or ArpackError
                continue

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.outputs_list.append(mesh_data)


    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        verts = self.verts_list[idx]
        faces = self.faces_list[idx]
        mesh_data = self.outputs_list[idx]
        # print(mesh_data["mesh_relative_path"])
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts, faces,
                                                                                           k_eig=self.k_eig,
                                                                                           op_cache_dir=self.op_cache_dir)
        return verts, faces, frames, mass, L, evals, evecs, gradX, gradY, mesh_data


def load_point_clouds_numpy(data_root_dir, num_points, label_names, filter_criteria=None, data_fraction=1.0):
    file_manager = MeshDatasetFileManager(data_root_dir)
    clouds, targets = file_manager.load_numpy_pointclouds(num_points, desired_label_names=label_names)
    p = np.random.permutation(len(clouds))
    clouds = clouds[p]
    targets = targets[p]
    num_files = len(clouds)
    num_file_to_use = int(data_fraction * num_files)

    print("Total number of point clouds processed: ", num_file_to_use)
    return clouds[:num_file_to_use, :, :], targets[:num_file_to_use, :]

def load_point_clouds_manual(data_root_dir, num_points, label_names, filter_criteria=None, use_augmentations=True, data_fraction=1.0):
    file_manager = MeshDatasetFileManager(data_root_dir)
    all_point_clouds = []
    all_label = []
    label_names = label_names

    # Open the base directory and get the contents
    data_files = file_manager.get_target_files(absolute=True)
    num_files = len(data_files)
    num_file_to_use = int(data_fraction * num_files)
    data_files = np.random.choice(data_files, size=num_file_to_use, replace=False)

    # Now parse through all the files
    for data_file in tqdm(data_files):
        # Load the file
        mesh, instances = file_manager.get_mesh_and_instances_from_target_file(data_file)
        if not use_augmentations:
            instances = [instances[0]]  # Only take the default instance

        for instance_data in instances:
            if not filter_criteria(mesh, instance_data):
                continue

            mesh = get_augmented_mesh(mesh, instance_data)
            vertices, _ = trimesh.sample.sample_surface(mesh, count=num_points)

            label = np.array([instance_data[label_name] for label_name in label_names])
            # if "centroid" in label_names:
            #     label = np.concatenate((np.mean(vertices, axis=0), )

            all_point_clouds.append(vertices)
            all_label.append(label)

    print("Total number of point clouds processed: ", len(all_point_clouds))
    point_clouds = np.stack(all_point_clouds, axis=0).astype('float32')
    label = np.stack(all_label, axis=0).astype('float32')

    return point_clouds, label

class PointCloudDataset(Dataset):
    def __init__(self, data_root_dir, num_points, label_names, partition='train', filter_criteria=None, use_augmentations=True, data_fraction=1.0, use_numpy=True, normalize=False):
        # filter_criteria of the form filter_criteria(mesh, instance_data)->boolean
        if use_numpy:
            print("Loading data from numpy...")
            if use_augmentations:
                print("Warning: augmentations are not being used")
            if filter_criteria is not None:
                    print("Warning: filters are not being used")
            if normalize:
                print("Note: Data is being normalized")
            self.point_clouds, self.label = load_point_clouds_numpy(data_root_dir=data_root_dir,
                                                                    num_points=2 * num_points,
                                                                    label_names=label_names,
                                                                    data_fraction=data_fraction)
        else:
            self.point_clouds, self.label = load_point_clouds_manual(data_root_dir=data_root_dir,
                                                                     num_points=2 * num_points,
                                                                     label_names=label_names,
                                                                     data_fraction=data_fraction,
                                                                     filter_criteria=filter_criteria,
                                                                     use_augmentations=use_augmentations)
        # Normalize each target
        if normalize:
            # std = np.std(self.label, axis=0)
            # mean = np.mean(self.label, axis=0)
            # self.label = (self.label - mean) / std

            # Normalize inputs
            # First get bounding boxes
            bound_max = np.max(self.point_clouds, axis=1)
            bound_min = np.min(self.point_clouds, axis=1)
            bound_length = np.max(bound_max - bound_min, axis=1) # maximum length
            # scale to box of 0, 1
            self.normalization_scale = 1.0/bound_length
            normalization_scale_multiplier = np.repeat(self.normalization_scale[:, np.newaxis], len(self.point_clouds[0]), axis=1)
            normalization_scale_multiplier = np.repeat(normalization_scale_multiplier[:, :, np.newaxis], 3, axis=2)
            self.normalization_order = 1
            self.point_clouds = normalization_scale_multiplier * self.point_clouds
            self.label = self.label * np.repeat(np.power(self.normalization_scale, self.normalization_order)[:, np.newaxis], len(self.label[0]), axis=1)


        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.point_clouds[item][:self.num_points] # Sample points from the mesh
        label = self.label[item] # Grab the output
        if self.partition == 'train':
            # pointcloud = dgcnn_data.translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.point_clouds.shape[0]