import json

import numpy as np
import potpourri3d as pp3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import diffusion_net
from dataset.process_and_save import MeshDatasetFileManager


class DiffusionNetDataset(Dataset):
    def __init__(self, data_root_dir, split_size, k_eig, exclude_dict=None, filter_criteria=None, op_cache_dir=None, data_fraction=1.0, label_names=None):
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
            with open(data_file, 'r') as f:
                target_master = json.load(f)
            mesh_path = file_manager.root_dir + target_master["mesh_relative_path"]

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
            # self.outputs_list.append(mesh_data)


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
