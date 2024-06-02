import util
from dataset.process_and_save import MeshDatasetFileManager, get_augmented_mesh
from torch.utils.data import Dataset
import diffusion_net
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from dataset.process_and_save_temp import DatasetManager, Augmentation
from typing import List
import json
import math
import trimesh
import torch
import trimesh_util

from diffusion_net.layers import DiffusionNet

class DiffusionNetDataset(Dataset):
    def __init__(self, data_root_dir, k_eig, outputs_at, augmentations: str | List[Augmentation] = "none",
                 op_cache_dir=None, data_fraction=1.0, label_names=None,
                 extra_vertex_label_names=None, extra_global_label_names=None,
                 augment_random_rotate=True, is_training=True, cache_operators=True):
        self.root_dir = data_root_dir
        self.k_eig = k_eig
        self.k_eig_list = []
        self.op_cache_dir = op_cache_dir
        self.outputs_at = outputs_at

        self.entries = {}

        self.augmentations = augmentations

        self.augment_random_rotate = augment_random_rotate
        self.is_training = is_training

        file_manager = DatasetManager(data_root_dir)

        self.all_faces = []
        self.all_vertices = []
        self.all_labels = []
        label_names = label_names

        # Open the base directory and get the contents
        num_files = len(file_manager.mesh_folder_names)
        num_file_to_use = int(data_fraction * num_files)
        mesh_folders = file_manager.get_mesh_folders(num_file_to_use)
        # mesh_folder_names = mesh_folder_names[util.get_permutation_for_list(mesh_folder_names, num_file_to_use)]

        # Now parse through all the files
        print("Loading Meshes")
        for mesh_folder in tqdm(mesh_folders):
            # TODO load default aug if desired
            if augmentations == "none":
                mesh_labels = [mesh_folder.load_default_mesh()]
            elif augmentations == "all":
                mesh_labels = mesh_folder.load_all_augmentations()
                # mesh_labels = mesh_folder.load_all_augmentations()
            else:
                print("Not Implemented")
                return
                # mesh_labels = mesh_folder.load_specific_augmentations_if_available(self.augmentations)

            for mesh_label in mesh_labels:
                mesh_data = mesh_label.convert_to_data(self.outputs_at, label_names,
                                                       extra_vertex_label_names=extra_vertex_label_names,
                                                       extra_global_label_names=extra_global_label_names)
                verts = mesh_data.vertices
                aug_verts = mesh_data.augmented_vertices
                label = mesh_data.labels
                faces = mesh_data.faces

                verts = torch.tensor(verts).float()
                faces = torch.tensor(faces)
                aug_verts = torch.tensor(aug_verts).float()
                label = torch.tensor(label).float()

                # Attempt to get eigen decomposition. If cannot, skip
                # try:
                if cache_operators:
                    diffusion_net.geometry.get_operators(verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
                # except:  # Or valueerror or ArpackError
                #     print("eigen error")
                #     continue

                self.all_faces.append(faces)
                self.all_vertices.append(aug_verts)
                self.all_labels.append(label)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        verts = self.all_vertices[idx]
        faces = self.all_faces[idx]
        label = self.all_labels[idx] # TODO convert to float here?

        # TODO data gets permuted after batch
        # Randomly rotate positions
        # if self.augment_random_rotate and self.is_training:
        #     verts = diffusion_net.utils.random_rotate_points(verts)

        return verts, faces, label

class DiffusionNetDataset2(Dataset):
    def __init__(self, data_root_dir, k_eig, filter_criteria=None,
                 op_cache_dir=None, data_fraction=1.0, label_names=None,
                 augment_random_rotate=True, is_training=True):
        self.root_dir = data_root_dir
        # self.split_size = split_size  # pass None to take all entries (except those in exclude_dict)
        self.k_eig = k_eig
        self.k_eig_list = []
        self.op_cache_dir = op_cache_dir

        self.entries = {}

        self.augment_random_rotate = augment_random_rotate
        self.is_training = is_training

        file_manager = MeshDatasetFileManager(data_root_dir)
        # self.all_meshes = []
        self.all_faces = []
        self.all_vertices = []
        self.all_labels = []
        label_names = label_names

        # Open the base directory and get the contents
        data_files = file_manager.get_target_files(absolute=True)
        num_files = len(data_files)
        num_file_to_use = int(data_fraction * num_files)
        data_files = np.random.choice(data_files, size=num_file_to_use, replace=False)

        # Now parse through all the files
        for data_file in tqdm(data_files):
            mesh, instances = file_manager.get_mesh_and_instances_from_target_file(data_file)
            for instance_data in instances:
                if not filter_criteria(mesh, instance_data):
                    continue

                mesh = get_augmented_mesh(mesh, instance_data)
                mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
                verts = torch.tensor(mesh_aux.vertices).float()
                faces = torch.tensor(mesh_aux.faces)

                # Attempt to get eigen decomposition. If cannot, skip
                try:
                    diffusion_net.geometry.get_operators(verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
                    # frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts, faces,
                    #                                                                                    k_eig=self.k_eig,
                    #                                                                                    op_cache_dir=self.op_cache_dir)
                except:  # Or valueerror or ArpackError
                    continue

                label = np.array([instance_data[label_name] for label_name in label_names])

                self.all_faces.append(faces)
                self.all_vertices.append(verts)
                self.all_labels.append(label)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        # mesh_data = self.all_meshes[idx]
        verts = self.all_vertices[idx]
        faces = self.all_faces[idx]
        label = self.all_labels[idx] # TODO convert to float here?

        # Already occuring
        # mesh_data.to(self.device)
        # TODO convert mesh data to device?
        # TODO data gets permuted after batch
        # Randomly rotate positions
        if self.augment_random_rotate and self.is_training:
            verts = diffusion_net.utils.random_rotate_points(verts)

        return verts, faces, label

class DiffusionNetWrapper(nn.Module):
    def __init__(self, model_args, op_cache_dir, device):
        super(DiffusionNetWrapper, self).__init__()

        input_feature_type = model_args["input_feature_type"]
        num_outputs = model_args["num_outputs"]
        C_width = model_args["C_width"]
        N_block = model_args["N_block"]
        last_activation = model_args["last_activation"]
        outputs_at = model_args["outputs_at"]
        mlp_hidden_dims = model_args["mlp_hidden_dims"]
        dropout = model_args["dropout"]
        with_gradient_features = model_args["with_gradient_features"]
        with_gradient_rotations = model_args["with_gradient_rotations"]
        diffusion_method = model_args["diffusion_method"]
        self.k_eig = model_args["k_eig"]
        self.device = device

        self.input_feature_type = input_feature_type
        self.op_cache_dir = op_cache_dir
        C_in = {'xyz': 3, 'hks': 16}[self.input_feature_type]

        self.wrapped_model = DiffusionNet(C_in=C_in, C_out=num_outputs, C_width=C_width, N_block=N_block,
                                          last_activation=last_activation, outputs_at=outputs_at,
                                          mlp_hidden_dims=mlp_hidden_dims, dropout=dropout,
                                          with_gradient_features=with_gradient_features,
                                          with_gradient_rotations=with_gradient_rotations,
                                          diffusion_method=diffusion_method)

    def forward(self, verts, faces):
        # TODO: this assumes batch size 1 right now
        # Calculate properties
        verts = verts[0]
        faces = faces[0]
        # raw_verts = verts[:, :3]
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts[:, :3], faces,
                                                                                           k_eig=self.k_eig,
                                                                                           op_cache_dir=self.op_cache_dir)
        verts = verts.to(self.device)
        faces = faces.to(self.device)
        # frames = frames.to(device)
        mass = mass.to(self.device)
        L = L.to(self.device)
        evals = evals.to(self.device)
        evecs = evecs.to(self.device)
        gradX = gradX.to(self.device)
        gradY = gradY.to(self.device)
        # Construct features
        if self.input_feature_type == 'xyz':
            features = verts
        else:  # self.input_feature_type == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)  # TODO autoscale here

        out = self.wrapped_model.forward(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX,
                                          gradY=gradY, faces=faces)
        return out[None, :, :]