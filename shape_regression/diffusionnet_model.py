from __future__ import annotations

from dataset.imbalanced_data_2 import ImbalancedWeightingNd, sample_equal_vertices_from_list
from torch.utils.data import Dataset
import diffusion_net
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from dataset.process_and_save_temp import DatasetManager, Augmentation
from typing import List
import torch
from dataset.categorical import CategoricalMap
import util
from dataset.VertexAggregator import VertexAggregator
from diffusion_net.layers import DiffusionNet
from diffusion_net.geometry import toNP
from dataset.imbalanced_data import get_imbalanced_weight_nd, non_outlier_indices, non_outlier_indices_vertices_nclass


def mesh_is_valid_diffusion(verts, faces):
    verts_np = toNP(verts).astype(np.float64)
    faces_np = toNP(faces)
    return not (faces_np > len(verts_np)).any()

def translate_pointcloud(pointcloud, device):
    # pointcloud is points x dim
    xyz_add = torch.zeros(pointcloud.size()[-1])
    xyz_add[:3] = torch.from_numpy(np.random.uniform(low=-0.1, high=0.1, size=[3]))

    return pointcloud + xyz_add.to(device)

class DiffusionNetDataset(Dataset):
    def __init__(self, data_root_dir, k_eig, args, partition, augmentations: str | List[Augmentation] = "none",
                 op_cache_dir=None, data_fraction=1.0, num_data=None,
                 augment_random_rotate=True, cache_operators=True, use_imbalanced_weights=False,
                 remove_outlier_ratio=0.0, aggregator: VertexAggregator=None):
        self.root_dir = data_root_dir
        self.k_eig = k_eig
        self.k_eig_list = []
        self.op_cache_dir = op_cache_dir
        if op_cache_dir is None:
            self.op_cache_dir = data_root_dir + "../op_cache/"
        self.outputs_at = args["outputs_at"]
        self.partition = partition
        self.augmentations = augmentations
        self.augment_random_rotate = augment_random_rotate

        self.aggregator = aggregator

        file_manager = DatasetManager(data_root_dir, partition=partition)

        self.all_faces = []
        self.all_vertices = []
        self.all_labels = []
        label_names = args["label_names"]

        # self.wiggle_position = wiggle_position

        # Open the base directory and get the contents
        num_files_to_use = num_data
        if num_files_to_use is None:
            num_files_to_use = int(data_fraction * file_manager.num_data)
        elif num_data > file_manager.num_data:
            num_files_to_use = file_manager.num_data

        mesh_folders = file_manager.get_mesh_folders(num_files_to_use)

        print("loading augmentations: ")
        print(self.augmentations)

        # Now parse through all the files
        print("Loading Meshes")
        outputs_at_to_load_from = self.outputs_at



        if aggregator is not None:
            outputs_at_to_load_from = "vertices"
        for mesh_folder in tqdm(mesh_folders):
            mesh_labels = mesh_folder.load_mesh_with_augmentations(self.augmentations)

            for mesh_label in mesh_labels:
                mesh_data = mesh_label.convert_to_data(outputs_at_to_load_from, label_names,
                                                       extra_vertex_label_names=args["input_append_vertex_label_names"],
                                                       extra_global_label_names=args["input_append_global_label_names"])

                verts = mesh_data.vertices
                aug_verts = mesh_data.augmented_vertices
                label = mesh_data.labels
                if self.aggregator is not None:
                    label = self.aggregator.aggregate(label)
                faces = mesh_data.faces

                if len(verts) > 1e6:
                    print("NOTE: dataset is removing nverts > 1e6")
                    continue

                tensor_vert = torch.tensor(verts).float()
                tensor_face = torch.tensor(faces)

                # Filters
                if not np.isfinite(verts).all() or not np.isfinite(faces).all():
                    print("Dataset: Nan found in mesh. skipping", mesh_folder.mesh_name, "Recommending manual deletion")
                    continue

                if not mesh_is_valid_diffusion(tensor_vert, tensor_face):
                    print("Dataset: Face index exceeds vertices. skipping", mesh_folder.mesh_name, "Recommending manual deletion")
                    continue

                if not np.isfinite(label).all():
                    print("Dataset: nan in labels. skipping", mesh_folder.mesh_name, "Recommending manual deletion")
                    continue

                # Attempt to get eigen decomposition. If cannot, skip
                if cache_operators:
                    try:
                        diffusion_net.geometry.get_operators(tensor_vert, tensor_face, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
                    except:  # Or valueerror or ArpackError
                        print("Dataset: Error calculating decomposition. Skipping", mesh_folder.mesh_name, "Recommending manual deletion")
                        continue

                self.all_faces.append(faces)
                self.all_vertices.append(aug_verts)
                self.all_labels.append(label)

        if remove_outlier_ratio > 0.0:
            print("Removing outliers")
            timer = util.Stopwatch()
            timer.start()
            if self.outputs_at == "global":
                keep_indices = non_outlier_indices(self.all_labels, num_bins=15,
                                                   threshold_ratio_to_remove=remove_outlier_ratio)
            else:  # vertices
                keep_indices = non_outlier_indices_vertices_nclass(self.all_labels, num_bins=15,
                                                                   threshold_ratio_to_remove=remove_outlier_ratio)
            original_length = len(self.all_labels)
            self.all_faces = [self.all_faces[i] for i in keep_indices]
            self.all_vertices = [self.all_vertices[i] for i in keep_indices]
            self.all_labels = [self.all_labels[i] for i in keep_indices]
            new_length = len(self.all_labels)
            print("Time to remove outliers:", timer.get_time())
            print("Removed", original_length - new_length, "outliers.")

        ## Classification
        self.categorical_thresholds = args["use_category_thresholds"]
        self.do_classification = False
        if self.categorical_thresholds is not None:
            cat_map = CategoricalMap(self.categorical_thresholds)
            self.all_labels = cat_map.to_category(self.all_labels)
            self.do_classification = True



        ## Imbalanced Weighting
        self.imbalanced_weighting = None
        if use_imbalanced_weights:
            if self.outputs_at == "vertices":
                # data x vertices x dim -> data*vertices x dim
                labels_concatenated = sample_equal_vertices_from_list(num_sample=256, data_list=self.all_labels)
                labels_concatenated = np.concatenate(labels_concatenated, axis=0)
                if len(labels_concatenated) > 10000:
                    rand_indices = util.get_permutation_for_list(labels_concatenated, 10000)
                    labels_concatenated = labels_concatenated[rand_indices]
                self.imbalanced_weighting = ImbalancedWeightingNd(labels_concatenated, do_classification=self.do_classification)
            else:
                # if global
                self.imbalanced_weighting = ImbalancedWeightingNd(self.all_labels, do_classification=self.do_classification)
        self.has_weights = False
        if self.imbalanced_weighting is not None:
            self.has_weights = True



    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        verts = self.all_vertices[idx]
        faces = self.all_faces[idx]
        label = self.all_labels[idx]

        # if self.partition == "train" and self.wiggle_position:
        #     verts = translate_pointcloud(verts)

        verts = torch.tensor(verts).float()
        faces = torch.tensor(faces)
        label = torch.tensor(label).float()



        # Randomly rotate positions
        # if self.augment_random_rotate and self.is_training:
        #     verts = diffusion_net.utils.random_rotate_points(verts)

        return verts, faces, label



class DiffusionNetWrapper(nn.Module):
    def __init__(self, model_args, op_cache_dir, device, augment_perturb_position=False):
        super(DiffusionNetWrapper, self).__init__()

        input_feature_type = model_args["input_feature_type"]
        num_outputs = model_args["num_outputs"]
        C_width = model_args["C_width"]
        N_block = model_args["N_block"]
        # last_activation = model_args["last_activation"]
        self.outputs_at = model_args["outputs_at"]
        output_at_to_pass = self.outputs_at
        if output_at_to_pass == "global":
            output_at_to_pass = "global_mean"
        mlp_hidden_dims = model_args["mlp_hidden_dims"]
        dropout = model_args["dropout"]
        with_gradient_features = model_args["with_gradient_features"]
        with_gradient_rotations = model_args["with_gradient_rotations"]
        diffusion_method = model_args["diffusion_method"]
        self.k_eig = model_args["k_eig"]
        self.device = device

        self.last_layer = None
        if "last_layer" in model_args:
            if model_args["last_layer"] == "sigmoid":
                self.last_layer = torch.nn.Sigmoid()
            elif model_args["last_layer"] == "softmax":
                self.last_layer = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.LogSoftmax(dim=-1))


        self.input_feature_type = input_feature_type
        self.op_cache_dir = op_cache_dir
        C_in = {'xyz': 3, 'hks': 16}[self.input_feature_type]

        self.augment_perturb_position = augment_perturb_position

        self.wrapped_model = DiffusionNet(C_in=C_in, C_out=num_outputs, C_width=C_width, N_block=N_block,
                                          outputs_at=output_at_to_pass,
                                          mlp_hidden_dims=mlp_hidden_dims, dropout=dropout,
                                          with_gradient_features=with_gradient_features,
                                          with_gradient_rotations=with_gradient_rotations,
                                          last_activation=self.last_layer,
                                          diffusion_method=diffusion_method)

    def forward(self, verts, faces):
        # TODO: this assumes batch size 1 right now
        # Calculate properties
        verts = verts[0]
        faces = faces[0]

        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts[:, :3], faces,
                                                                                       k_eig=self.k_eig,
                                                                                       op_cache_dir=self.op_cache_dir)
        if self.augment_perturb_position:
            verts = translate_pointcloud(verts, self.device)

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
            # TODO Append extra labels here if chosen to do so

        out = self.wrapped_model.forward(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX,
                                          gradY=gradY, faces=faces)
        out = torch.nan_to_num(out) # TODO is this the best option?
        if self.outputs_at == "vertices":
            return out[None, :, :]
        else:
            return out[None, :]