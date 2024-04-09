import numpy as np
import trimesh
from torch.utils.data import Dataset
from tqdm import tqdm
from dataset.process_and_save import MeshDatasetFileManager, get_augmented_mesh
from dataset.imbalanced_data import get_imbalanced_weight_nd

def load_point_clouds_numpy(data_root_dir, num_points, label_names,
                            data_fraction=1.0, sampling_method="mixed", outputs_at="global"):
    file_manager = MeshDatasetFileManager(data_root_dir)
    clouds, targets = file_manager.load_numpy_pointclouds(num_points, desired_label_names=label_names,
                                                          sampling_method=sampling_method, outputs_at=outputs_at)
    p = np.random.permutation(len(clouds))
    clouds = clouds[p]
    targets = targets[p]
    num_files = len(clouds)
    num_file_to_use = int(data_fraction * num_files)

    print("Total number of point clouds processed: ", num_file_to_use)
    return clouds[:num_file_to_use], targets[:num_file_to_use]

def load_point_clouds_manual(data_root_dir, num_points, label_names, filter_criteria=None, use_augmentations=True,
                             data_fraction=1.0, sampling_method="mixed"):
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
            vertices, _ = trimesh.sample.sample_surface(mesh, count=num_points, face_weight=sampling_method)

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
    def __init__(self, data_root_dir, num_points, label_names, partition='train', filter_criteria=None, outputs_at="global",
                 use_augmentations=True, data_fraction=1.0, use_numpy=True, normalize=False, sampling_method="mixed",
                 imbalance_weight_num_bins=1, normalize_outputs=False):
        # filter_criteria of the form filter_criteria(mesh, instance_data)->boolean
        self.outputs_at = outputs_at
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
                                                                    data_fraction=data_fraction,
                                                                    sampling_method=sampling_method,
                                                                    outputs_at=outputs_at)
        else:
            if outputs_at == "vertices":
                print("Warning: outputs_at has not been implemented for vertices with this loading method")
            self.point_clouds, self.label = load_point_clouds_manual(data_root_dir=data_root_dir,
                                                                     num_points=2 * num_points,
                                                                     label_names=label_names,
                                                                     data_fraction=data_fraction,
                                                                     filter_criteria=filter_criteria,
                                                                     use_augmentations=use_augmentations,
                                                                     sampling_method=sampling_method)
        # Normalize each target
        if normalize:
            # Normalize inputs
            # First get bounding boxes
            bound_max = np.max(self.point_clouds, axis=1)
            bound_min = np.min(self.point_clouds, axis=1)
            centroid = np.mean(self.point_clouds, axis=1)
            bound_length = np.max(bound_max - bound_min, axis=1) # maximum length
            # scale to box of 0, 1
            self.normalization_scale = 1.0/bound_length
            normalization_scale_multiplier = np.repeat(self.normalization_scale[:, np.newaxis], len(self.point_clouds[0]), axis=1)
            normalization_scale_multiplier = np.repeat(normalization_scale_multiplier[:, :, np.newaxis], 3, axis=2)

            centering = np.repeat(centroid[:, np.newaxis, :], num_points*2, axis=1)
            self.point_clouds -= centering
            self.point_clouds = normalization_scale_multiplier * self.point_clouds

            self.normalization_order = 3
            self.label = self.label * np.repeat(np.power(self.normalization_scale, self.normalization_order)[:, np.newaxis], len(self.label[0]), axis=1)

        if normalize_outputs:
            length = np.max(self.label, axis=0) - np.min(self.label, axis=0)
            center = np.mean(self.label, axis=0)
            self.label = (self.label - center) / length

        self.weights = None
        if imbalance_weight_num_bins > 1:
            self.weights = get_imbalanced_weight_nd(self.label, num_bins=imbalance_weight_num_bins, modifier=None)
            # self.weights = np.sqrt(self.weights)

        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.point_clouds[item]
        label = self.label[item]
        weight = 1
        if self.weights is not None:
            weight = self.weights[item]

        # Grab random num_points from the point cloud
        if self.partition == 'train':
            p = np.random.permutation(len(pointcloud))[:self.num_points]
            # pointcloud = dgcnn_data.translate_pointcloud(self.pointcloud)
        else:
            p = np.arange(self.num_points)

        if self.outputs_at == "vertices":
            label = label[p]
            # label = np.transpose(label, (1, 0))
        pointcloud = pointcloud[p]
        # pointcloud = np.transpose(pointcloud, (1, 0))

        return pointcloud, label, weight

    def __len__(self):
        return self.point_clouds.shape[0]