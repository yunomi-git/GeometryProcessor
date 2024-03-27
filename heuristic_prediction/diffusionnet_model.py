from dataset.process_and_save import MeshDatasetFileManager, get_augmented_mesh
from torch.utils.data import Dataset
import diffusion_net
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import json
import math
import trimesh
import torch
import trimesh_util

from diffusion_net.layers import DiffusionNet


class DiffusionNetData:
    def __init__(self, verts, faces, frames, mass, L, evals, evecs, gradX, gradY):
        self.verts = verts
        self.faces = faces
        self.frames = frames
        self.mass = mass
        self.L = L
        self.evals = evals
        self.evecs = evecs
        self.gradX = gradX
        self.gradY = gradY

    def to(self, device):
        self.verts = self.verts.to(device)
        self.faces = self.faces.to(device)
        self.frames = self.frames.to(device)
        self.mass = self.mass.to(device)
        self.L = self.L.to(device)
        self.evals = self.evals.to(device)
        self.evecs = self.evecs.to(device)
        self.gradX = self.gradX.to(device)
        self.gradY = self.gradY.to(device)

class DiffusionNetDataset(Dataset):
    def __init__(self, data_root_dir, split_size, k_eig, filter_criteria=None,
                 op_cache_dir=None, data_fraction=1.0, label_names=None, augment_random_rotate=True, is_training=True):
        self.file_manager = MeshDatasetFileManager(data_root_dir)
        self.root_dir = data_root_dir
        self.split_size = split_size  # pass None to take all entries (except those in exclude_dict)
        self.k_eig = k_eig
        self.k_eig_list = []
        self.op_cache_dir = op_cache_dir

        self.entries = {}

        self.augment_random_rotate = augment_random_rotate
        self.is_training = is_training

        file_manager = MeshDatasetFileManager(data_root_dir)
        self.all_meshes = []
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
                faces = torch.tensor(mesh_aux.facets)

                # Attempt to get eigen decomposition. If cannot, skip
                try:
                    # diffusion_net.geometry.get_operators(verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
                    frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts, faces,
                                                                                                       k_eig=self.k_eig,
                                                                                                       op_cache_dir=self.op_cache_dir)
                    mesh_data = DiffusionNetData(verts=verts, faces=faces, frames=frames,
                                                 mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
                except:  # Or valueerror or ArpackError
                    continue

                label = np.array([instance_data[label_name] for label_name in label_names])

                # center and unit scale
                # verts = diffusion_net.geometry.normalize_positions(verts)

                self.all_meshes.append(mesh_data)
                self.all_labels.append(label)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        mesh_data = self.all_meshes[idx]
        label = self.all_labels[idx] # TODO convert to float here?

        # Already occuring
        # mesh_data.to(self.device)
        # TODO convert mesh data to device?
        # Randomly rotate positions
        if self.augment_random_rotate and self.is_training:
            mesh_data.verts = diffusion_net.utils.random_rotate_points(mesh_data.verts)

        return mesh_data, label

class DiffusionNetWrapper(nn.Module):
    def __init__(self, input_feature_type, num_outputs, C_width=128, N_block=4, last_activation=None,
                 outputs_at='vertices', mlp_hidden_dims=None, dropout=True, with_gradient_features=True,
                 with_gradient_rotations=True, diffusion_method='spectral'):

        self.input_feature_type = input_feature_type
        C_in = {'xyz': 3, 'hks': 16}[self.input_feature_type]

        self.wrapped_model = DiffusionNet(C_in=C_in, C_out=num_outputs, C_width=C_width, N_block=N_block,
                                          last_activation=last_activation, outputs_at=outputs_at,
                                          mlp_hidden_dims=mlp_hidden_dims, dropout=dropout,
                                          with_gradient_features=with_gradient_features,
                                          with_gradient_rotations=with_gradient_rotations,
                                          diffusion_method=diffusion_method)

    def forward(self, x: DiffusionNetData):
        # Construct features
        if self.input_feature_type == 'xyz':
            features = x.verts
        else:  # self.input_feature_type == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(x.evals, x.evecs, 16)  # TODO autoscale here

        return self.wrapped_model.forward(features, x.mass, L=x.L, evals=x.evals, evecs=x.evecs, gradX=x.gradX,
                                          gradY=x.gradY, faces=x.faces)
