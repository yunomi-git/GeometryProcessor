import shutil
import os
import sys
import random
import numpy as np
import json
import math
import trimesh
import trimesh_util

import paths
import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d

import diffusion_net
from diffusion_net.utils import toNP
from tqdm import tqdm

from dataset.process_and_save import MeshDatasetFileManager, get_augmented_mesh
import dgcnn_net.data as dgcnn_data


class CachedMeshList:
    def init(self, data_root_dir, preprocessed=False):
        # Preprocess: was this information already saved
        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.outputs_list = []

        # Open the base directory and get the contents
        file_manager = MeshDatasetFileManager(data_root_dir)
        data_files = file_manager.get_target_files(absolute=True)

        # Now parse through all the files
        for data_file in tqdm(data_files):
            # print(data_file)
            # Load the file
            file_path = data_file
            with open(file_path, 'r') as f:
                mesh_data = json.load(f)

            # if mesh_data["vertices"] > 1e4:
            #     continue
            if mesh_data["vertices"] < 1e2:
                continue
            if math.isnan(mesh_data["thickness_violation"]):
                continue
            mesh_path = data_root_dir + mesh_data["mesh_relative_path"]

            verts, faces = pp3d.read_mesh(mesh_path)
            verts = torch.tensor(verts).float()
            faces = torch.tensor(faces)

            # center and unit scale
            # verts = diffusion_net.geometry.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.outputs_list.append(mesh_data)


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


class DGCNNDataSet(Dataset):
    def __init__(self, data_root_dir, num_points, partition='train', filter_criteria=None, use_augmentations=True):
        # filter_invalid of the form filter_invalid(mesh, instance_data)->boolean

        self.file_manager = MeshDatasetFileManager(data_root_dir)
        self.root_dir = data_root_dir
        all_point_clouds = []
        all_label = []

        # Open the base directory and get the contents
        data_files = self.file_manager.get_target_files(absolute=True)

        # Now parse through all the files
        for data_file in tqdm(data_files):
            # Load the file
            mesh, instances = self.file_manager.get_mesh_and_instances_from_target_file(data_file)
            if not use_augmentations:
                instances = [instances[0]] # Only take the default instance

            for instance_data in instances:
                if not filter_criteria(mesh, instance_data):
                    continue


                mesh = get_augmented_mesh(mesh, instance_data)
                mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
                vertices, _ = trimesh.sample.sample_surface(mesh, count=8192)

                # For label, only take the 4 metrics.
                # Overhang, Stairstep, Thicknesses, Gaps,
                label = np.array([instance_data["overhang_violation"],
                                  instance_data["stairstep_violation"],
                                  instance_data["thickness_violation"],
                                  instance_data["gap_violation"]])

                all_point_clouds.append(vertices)
                all_label.append(label)

        self.point_clouds = np.stack(all_point_clouds, axis=0).astype('float32')
        self.label = np.stack(all_label, axis=0).astype('float32')
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