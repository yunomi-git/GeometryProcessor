import numpy as np
import trimesh
from torch.utils.data import Dataset
from tqdm import tqdm
from dataset.imbalanced_data_2 import ImbalancedWeightingKde, ImbalancedWeightingNd
from dataset.categorical import CategoricalMap

import util
from dataset.process_and_save import MeshDatasetFileManager, get_augmented_mesh
from dataset.imbalanced_data import get_imbalanced_weight_nd, non_outlier_indices, non_outlier_indices_vertices_nclass
from dataset.process_and_save_temp import DatasetManager
def load_point_clouds_numpy(data_root_dir, num_points, label_names, append_label_names=None,
                            data_fraction=1.0, sampling_method="mixed", outputs_at="global"):
    file_manager = MeshDatasetFileManager(data_root_dir)
    clouds, targets, append_labels = file_manager.load_numpy_pointclouds(num_points,
                                                                         desired_label_names=label_names,
                                                                         extra_label_names=append_label_names,
                                                                         sampling_method=sampling_method,
                                                                         outputs_at=outputs_at)
    p = np.random.permutation(len(clouds))
    clouds = clouds[p]
    targets = targets[p]
    num_files = len(clouds)
    num_file_to_use = int(data_fraction * num_files)

    # combine clouds and extra labels
    # clouds are data x points x dim. labels are data x dim or data x points x dim
    if append_labels is not None:
        append_labels = append_labels[p]
        if outputs_at == "global":
            append_labels = np.repeat(append_labels[:, np.newaxis, :], repeats=len(clouds.shape[1]))
            clouds = np.append(clouds, append_labels, axis=2)
        else:
            clouds = np.append(clouds, append_labels, axis=2)
        print("Clouds have appended information. Shape: ", clouds.shape)

    print("Total number of point clouds processed: ", num_file_to_use)
    return clouds[:num_file_to_use], targets[:num_file_to_use]


def translate_pointcloud(pointcloud):
    # xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    # xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    # xyz_mult = np.ones(pointcloud.shape[1])
    # xyz_mult[:3] = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz_add = np.zeros(pointcloud.shape[1])
    xyz_add[:3] = np.random.uniform(low=-0.2, high=0.2, size=[3])

    # translated_pointcloud = np.add(np.multiply(pointcloud, xyz_mult), xyz_add).astype('float32')
    translated_pointcloud = np.add(pointcloud, xyz_add).astype('float32')

    return translated_pointcloud



class PointCloudDataset(Dataset):
    def __init__(self, data_root_dir, num_points, label_names, partition='train', outputs_at="global",
                 data_fraction=1.0, num_data=None, append_label_names=None, augmentations="all",
                 use_imbalanced_weights=False, remove_outlier_ratio=0.05, categorical_thresholds=None):
        self.outputs_at = outputs_at
        timer = util.Stopwatch()
        timer.start()

        dataset_manager = DatasetManager(dataset_path=data_root_dir, partition=partition)

        num_clouds = num_data
        if num_clouds is None:
            num_clouds = int(data_fraction * dataset_manager.num_data)
        elif num_clouds > dataset_manager.num_data:
            num_clouds = dataset_manager.num_data

        _, self.point_clouds, self.label = dataset_manager.load_numpy_pointcloud(
            num_clouds=num_clouds,
            num_points=2 * num_points,
            augmentations=augmentations,
            outputs_at=outputs_at,
            desired_label_names=label_names,
            extra_vertex_label_names=append_label_names,
            extra_global_label_names=None)

        print("Num data loaded: " + str(len(self.point_clouds)))

        if remove_outlier_ratio > 0.0:
            if outputs_at == "global":
                keep_indices = non_outlier_indices(self.label, num_bins=15,
                                                   threshold_ratio_to_remove=remove_outlier_ratio)
            else: # vertices
                keep_indices = non_outlier_indices_vertices_nclass(self.label, num_bins=15,
                                                                   threshold_ratio_to_remove=remove_outlier_ratio)
            original_length = len(self.label)
            self.point_clouds = self.point_clouds[keep_indices]
            self.label = self.label[keep_indices]
            new_length = len(self.label)
            print("Removed", original_length - new_length, "outliers.")

        self.categorical_thresholds = categorical_thresholds
        if self.categorical_thresholds is not None:
            cat_map = CategoricalMap(categorical_thresholds)
            self.label = cat_map.to_category(self.label)

        self.imbalanced_weighting = None
        self.has_weights = use_imbalanced_weights
        if use_imbalanced_weights:
            print("Calculating imbalanced weights")
            if self.outputs_at == "vertices":
                # data x vertices x dim -> data*vertices x dim
                labels_concatenated = np.concatenate(self.label[:, :256, :], axis=0)
                if len(labels_concatenated) > 10000:
                    rand_indices = util.get_permutation_for_list(labels_concatenated, 10000)
                    labels_concatenated = labels_concatenated[rand_indices]
                self.imbalanced_weighting = ImbalancedWeightingNd(labels_concatenated)
            else:
                # if global
                self.imbalanced_weighting = ImbalancedWeightingNd(self.label)

        print("Time to load data: ", timer.get_time())
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.point_clouds[item]
        label = self.label[item]

        # Grab random num_points from the point cloud
        if self.partition == 'train':
            p = np.random.permutation(len(pointcloud))[:self.num_points]
            # pointcloud = translate_pointcloud(pointcloud)
        else:
            p = np.arange(self.num_points)

        if self.outputs_at == "vertices":
            label = label[p]
        pointcloud = pointcloud[p]

        return pointcloud, label

    def __len__(self):
        return self.point_clouds.shape[0]










# def load_point_clouds_manual(data_root_dir, num_points, label_names, filter_criteria=None, use_augmentations=True,
#                              data_fraction=1.0, sampling_method="mixed"):
#     file_manager = MeshDatasetFileManager(data_root_dir)
#     all_point_clouds = []
#     all_label = []
#     label_names = label_names
#
#     # Open the base directory and get the contents
#     data_files = file_manager.get_target_files(absolute=True)
#     num_files = len(data_files)
#     num_file_to_use = int(data_fraction * num_files)
#     data_files = np.random.choice(data_files, size=num_file_to_use, replace=False)
#
#     # Now parse through all the files
#     for data_file in tqdm(data_files):
#         # Load the file
#         mesh, instances = file_manager.get_mesh_and_instances_from_target_file(data_file)
#         if not use_augmentations:
#             instances = [instances[0]]  # Only take the default instance
#
#         for instance_data in instances:
#             if not filter_criteria(mesh, instance_data):
#                 continue
#
#             mesh = get_augmented_mesh(mesh, instance_data)
#             vertices, _ = trimesh.sample.sample_surface(mesh, count=num_points, face_weight=sampling_method)
#
#             label = np.array([instance_data[label_name] for label_name in label_names])
#
#             all_point_clouds.append(vertices)
#             all_label.append(label)
#
#     print("Total number of point clouds processed: ", len(all_point_clouds))
#     point_clouds = np.stack(all_point_clouds, axis=0).astype('float32')
#     label = np.stack(all_label, axis=0).astype('float32')
#
#     return point_clouds, label