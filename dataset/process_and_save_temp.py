from __future__ import annotations

import paths
import trimesh_util
import trimesh
import numpy as np
from pathlib import Path
import json
from typing import List
import os
import shutil
import dataset.FolderManager as FolderManager
from tqdm import tqdm
import util
import argparse
from dataset.create_splits_data import load_mesh_folders_from_split_file, create_splits_file

# Grab back. From Project: rsync -uPr nyu@txe1-login.mit.edu:~/Datasets/DrivAerNet/Simplified_Remesh Datasets/DrivAerNet/Simplified_Remesh/


def get_submesh_index(mesh_name):
    index = mesh_name.rfind("_")
    if index == -1:
        return 0
    postscript = mesh_name[index + 1:] # TODO this may not be the final one

    return int(postscript)

class Augmentation:
    def __init__(self, rotation: np.ndarray, scale: np.ndarray):
        self.rotation = rotation
        self.scale = scale

    def as_json(self):
        return {
            "orientation": {"x": self.rotation[0], "y": self.rotation[1], "z": self.rotation[2]},
            "scale": {"x": self.scale[0], "y": self.scale[1], "z": self.scale[2]}
        }

    def as_string(self):
        simplify_number = lambda x: str(int(x * 100))
        return ("r" + simplify_number(self.rotation[0]) + "_" +
                simplify_number(self.rotation[1]) + "_" +
                simplify_number(self.rotation[2]) + "_" +
                "s" + simplify_number(self.scale[0]) + "_" +
                simplify_number(self.scale[1]) + "_" +
                simplify_number(self.scale[2]))


DEFAULT_AUGMENTATION = Augmentation(scale=np.array([1.0, 1.0, 1.0]), rotation=np.array([0.0, 0.0, 0.0]))

class MeshRawLabels:
    def __init__(self, vertices, faces, vertex_labels, global_labels, vertex_label_names, global_label_names):
        self.vertices = vertices
        self.faces = faces
        self.vertex_labels = vertex_labels
        self.global_labels = global_labels
        self.vertex_label_names = vertex_label_names
        self.global_label_names = global_label_names

    def get_vertex_labels(self, desired_vertex_label_names):
        label_indices = []
        for desired_label in desired_vertex_label_names:
            if desired_label not in self.vertex_label_names:
                print("Vertex label not found. available label names: ", self.vertex_label_names)
            label_indices.append(self.vertex_label_names.index(desired_label))
        return self.vertex_labels[:, label_indices]

    def get_global_labels(self, desired_global_label_names):
        label_indices = []
        for desired_label in desired_global_label_names:
            if desired_label not in self.global_label_names:
                print("Global label not found. available label names: ", self.global_label_names)
            label_indices.append(self.global_label_names.index(desired_label))
        return self.global_labels[label_indices]

    def convert_to_data(self, outputs_at, label_names,
                        extra_vertex_label_names=None, extra_global_label_names=None):
        augmented_vertices = self.vertices
        if extra_vertex_label_names is not None and len(extra_vertex_label_names) > 0:
            extra_vertex_labels = self.get_vertex_labels(extra_vertex_label_names)
            augmented_vertices = np.concatenate([augmented_vertices, extra_vertex_labels], axis=1)
        if extra_global_label_names is not None and len(extra_global_label_names) > 0:
            extra_global_labels = self.get_global_labels(extra_global_label_names)
            augmented_vertices = np.concatenate([augmented_vertices,
                                          np.repeat(extra_global_labels[np.newaxis, :], len(augmented_vertices))], axis=1)

        # augmented_vertices = np.stack(vertices, extra_vertex_labels, np.repeat(extra_global_labels[np.newaxis, :], len(vertices))) # TODO
        if outputs_at == "vertices":
            labels = self.get_vertex_labels(label_names)
        else:
            labels = self.get_global_labels(label_names)
        faces = self.faces
        if faces is not None:
            faces = faces.astype(np.int32)
        return GeometryReadyData(vertices=self.vertices.astype(np.float32),
                                 augmented_vertices=augmented_vertices.astype(np.float32),
                                 faces=faces,
                                 labels=labels.astype(np.float32))


def calculate_mesh_labels(mesh, augmentation: Augmentation=None, only_points=False, num_points=8192) -> MeshRawLabels:
    if augmentation is not None:
        mesh = trimesh_util.get_transformed_mesh_trs(mesh,
                                                     scale=augmentation.scale,
                                                     orientation=augmentation.rotation)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    if only_points:
        vertices, normals = mesh_aux.sample_and_get_normals(count=num_points*2)
        vertices = vertices[:num_points]
        normals = normals[:num_points]
        faces = None
    else:
        vertices = mesh_aux.vertices
        faces = mesh_aux.faces
        normals = mesh_aux.vertex_normals
    num_vertices = len(vertices)

    vertex_label_names = ["Thickness",
                          "Gaps",
                          "nx",
                          "ny",
                          "nz",
                          "curvature"]
    global_label_names = ["SurfaceArea",
                          "Volume",
                          "cx",
                          "cy",
                          "cz"]


    _, thicknesses, vertex_ids = mesh_aux.calculate_thickness_at_points(points=vertices,
                                                                         normals=normals,
                                                                         return_num_samples=False,
                                                                         return_ray_ids=True)
    if len(thicknesses) != num_vertices:
        # Attempt to repair
        print("WARNING: Repairing thicknesses")
        thicknesses = trimesh_util.repair_missing_mesh_values(mesh, vertex_ids, values=thicknesses, max_iterations=2)

    if len(thicknesses) != num_vertices:
        print("ERROR not all vertices returned thicknesses")
        return

    _, gaps, vertex_ids = mesh_aux.calculate_gaps_at_points(points=vertices,
                                                            normals=normals,
                                                            return_num_samples=False,
                                                            return_ray_ids=True)

    if len(gaps) != num_vertices:
        # Attempt to repair
        print("WARNING: Repairing gaps")
        thicknesses = trimesh_util.repair_missing_mesh_values(mesh, vertex_ids, values=gaps, max_iterations=2)

    if len(gaps) != num_vertices:
        print("ERROR not all vertices returned gaps")
        return

    _, curvatures, num_samples = mesh_aux.calculate_curvature_at_points(origins=vertices,
                                                                        face_ids=None,
                                                                        curvature_method="abs",
                                                                        use_abs=False,
                                                                        return_num_samples=True)
    if num_samples != num_vertices:
        print("ERROR not all vertices returned curvature")
        return

    surface_area = mesh_aux.surface_area
    volume = mesh_aux.volume
    centroid = np.mean(vertices, axis=0)

    vertex_labels = [thicknesses, gaps, normals[:, 0], normals[:, 1], normals[:, 2], curvatures]
    global_labels = [surface_area, volume, centroid[0], centroid[1], centroid[2]]

    vertex_labels = np.stack(vertex_labels).T
    global_labels = np.array(global_labels)

    return MeshRawLabels(vertices=vertices, faces=faces, vertex_labels=vertex_labels,
                         global_labels=global_labels, vertex_label_names=vertex_label_names,
                         global_label_names=global_label_names)

class MeshFolder:
    def __init__(self, dataset_path, mesh_name, only_points=False, num_points=8192):
        self.dataset_path = dataset_path
        self.mesh_name = mesh_name
        self.mesh_dir_path = dataset_path + str(mesh_name) + "/"
        self.mesh_stl_path = self.mesh_dir_path + "mesh.stl"

        self.only_points = only_points
        self.num_points = num_points

    def initialize_folder(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.mesh.export(self.mesh_dir_path + "mesh.stl")
        # self.num_vertices = len(self.mesh.vertices)
        # self.num_faces = len(self.mesh.faces)

    def delete_augmentation(self, augmentation: Augmentation):
        shutil.rmtree(self.mesh_dir_path + augmentation.as_string())

    def calculate_and_save_augmentation(self, augmentation: Augmentation, override):
        # First check if augmentation already exists
        if self.augmentation_is_cached(augmentation):
            if override:
                print("Overriding Augmentation")
                self.delete_augmentation(augmentation)
            else:
                print("Augmentation already exists. No action taken")
                return

        mesh = trimesh.load(self.mesh_dir_path + "mesh.stl")

        # Calculate
        mesh_labels = calculate_mesh_labels(mesh=mesh, augmentation=augmentation,
                                            only_points=self.only_points, num_points=self.num_points)

        if mesh_labels is None:
            return

        # Save
        aug_dir_path = self.mesh_dir_path + augmentation.as_string()
        Path(aug_dir_path).mkdir(parents=True, exist_ok=True)
        manifest = {
            "augmentation": augmentation.as_json(),
            "vertex_label_names": mesh_labels.vertex_label_names,
            "global_label_names": mesh_labels.global_label_names
        }
        with open(aug_dir_path + "/" + "manifest.json", 'w') as f:
            json.dump(manifest, f)
        Path(aug_dir_path).mkdir(parents=True, exist_ok=True)
        np.save(aug_dir_path + "/" + "vertices", mesh_labels.vertices)
        if mesh_labels.faces is not None:
            np.save(aug_dir_path + "/" + "faces", mesh_labels.faces)
        np.save(aug_dir_path + "/" + "vertex_labels", mesh_labels.vertex_labels)
        np.save(aug_dir_path + "/" + "global_labels", mesh_labels.global_labels)

    def augmentation_is_cached(self, augmentation):
        directory_manager = FolderManager.DirectoryPathManager(base_path=self.mesh_dir_path, base_unit_is_file=False)
        available_augmentations = directory_manager.get_file_names(extension=False)
        if isinstance(augmentation, Augmentation):
            return augmentation.as_string() in available_augmentations
        else: # (string)
            return augmentation in available_augmentations


    def load_mesh_with_augmentation(self, augmentation) -> MeshRawLabels:
        if isinstance(augmentation, Augmentation):
            aug_path = self.mesh_dir_path + augmentation.as_string() + "/"
        else: # (string)
            aug_path = self.mesh_dir_path + augmentation + "/"

        with open(aug_path + "/" + "manifest.json", 'r') as f:
            try:
                manifest = json.load(f)
            except:
                print(self.mesh_name)
                return
        vertex_label_names = manifest["vertex_label_names"]
        global_label_names = manifest["global_label_names"]
        vertices = np.load(aug_path + "vertices.npy")
        if os.path.exists(aug_path + "faces.npy"):
            faces = np.load(aug_path + "faces.npy")
        else:
            faces = None
        vertex_labels = np.load(aug_path + "vertex_labels.npy")
        global_labels = np.load(aug_path + "global_labels.npy")

        # TODO: Meshes were saved incorrectly the first time. need to redo
        return MeshRawLabels(vertices=vertices, faces=faces, vertex_labels=vertex_labels,
                             global_labels=global_labels, vertex_label_names=vertex_label_names,
                             global_label_names=global_label_names)

        # return MeshRawLabels(vertices=vertices, faces=faces, vertex_labels=vertex_labels.T,
        #                      global_labels=global_labels[0], vertex_label_names=vertex_label_names,
        #                      global_label_names=global_label_names)

    def load_mesh_with_augmentations(self, augmentations: List[Augmentation] | List[str]) -> List[MeshRawLabels]:
        if augmentations == "none" or augmentations is None:
            mesh_labels = [self.load_default_mesh()]
        elif augmentations == "all":
            mesh_labels = self.load_all_augmentations()
        else:
            mesh_labels = self.load_specific_augmentations_if_available(augmentations)
        return mesh_labels

    def load_default_mesh(self):
        return self.load_mesh_with_augmentation(DEFAULT_AUGMENTATION)

    def load_all_augmentations(self) -> List[MeshRawLabels]:
        directory_manager = FolderManager.DirectoryPathManager(base_path=self.mesh_dir_path, base_unit_is_file=False)
        available_augmentations = directory_manager.get_file_names(extension=False)
        mesh_labels = []
        for aug_string in available_augmentations:
            mesh_labels.append(self.load_mesh_with_augmentation(aug_string))
        return mesh_labels
        # return self.load_specific_augmentations_if_available(available_augmentations)
        # TODO This takes filenames not actual augmentations

    def load_specific_augmentations_if_available(self, augmentations: List[Augmentation]):
        mesh_labels = []
        for augmentation in augmentations:
            if self.augmentation_is_cached(augmentation):
                mesh_labels.append(self.load_mesh_with_augmentation(augmentation))
        return mesh_labels


class GeometryReadyData:
    # This is the data that gets loaded
    def __init__(self, vertices, augmented_vertices, labels, faces=None):
        # Inputs
        self.vertices = vertices
        self.faces = faces
        self.augmented_vertices = augmented_vertices
        # Outputs
        self.labels = labels

    def is_cloud(self):
        return self.faces is None

class DatasetManager:
    def __init__(self, dataset_path, split_file=None, partition=None):
        # if split file is being used, partition must be set to "test" or "train"
        self.dataset_path = dataset_path
        # get mesh folders
        if split_file is None:
            self.mesh_folder_names = os.listdir(self.dataset_path)
            self.mesh_folder_names.sort()
        else:
            self.mesh_folder_names = load_mesh_folders_from_split_file(split_file, partition)
        self.num_data = len(self.mesh_folder_names)

    def get_mesh_folders(self, num_meshes) -> List[MeshFolder]:#, num_meshes, augmentations: List[Augmentation]):
        # grab num_meshes random meshes
        mesh_folders = []
        mesh_names = util.get_random_n_in_list(self.mesh_folder_names, num_meshes)
        for mesh_name in mesh_names:
            mesh_folder = MeshFolder(self.dataset_path, mesh_name)
            mesh_folders.append(mesh_folder)

        return mesh_folders

    def load_mesh_data(self, num_meshes, augmentations: List[Augmentation], outputs_at="global",
                       desired_label_names=None, extra_vertex_label_names=None, extra_global_label_names=None) -> List[GeometryReadyData]:
        mesh_data_list = []

        # Now parse through all the files
        print("Loading Meshes")
        print("- Using augmentations: ")
        print(augmentations)

        mesh_folders = self.get_mesh_folders(num_meshes)
        for mesh_folder in tqdm(mesh_folders):
            mesh_labels = mesh_folder.load_mesh_with_augmentations(augmentations)
            for mesh_label in mesh_labels:
                mesh_data = mesh_label.convert_to_data(outputs_at, desired_label_names,
                                                       extra_vertex_label_names=extra_vertex_label_names,
                                                       extra_global_label_names=extra_global_label_names)

                if not np.isfinite(mesh_data.vertices).all() or not np.isfinite(mesh_data.faces).all():
                    print("Dataset: Nan found in mesh. skipping", mesh_folder.mesh_name, "Recommending manual deletion")
                    continue

                if not np.isfinite(mesh_data.labels).all():
                    print("Dataset: nan in labels. skipping", mesh_folder.mesh_name, "Recommending manual deletion")
                    continue

                mesh_data_list.append(mesh_data)
        return mesh_data_list
    def load_pointcloud_data(self, num_clouds, num_points, augmentations: List[Augmentation] | str, outputs_at="global",
                               desired_label_names=None, extra_vertex_label_names=None, extra_global_label_names=None):
        cloud_data_list = []

        # Now parse through all the files
        print("Loading Meshes")
        print("- Using augmentations: ")
        print(augmentations)

        mesh_folders = self.get_mesh_folders(num_clouds)
        for mesh_folder in tqdm(mesh_folders):
            mesh_labels = mesh_folder.load_mesh_with_augmentations(augmentations)
            # Ensure number of points is valid
            if len(mesh_labels[0].vertices) < num_points:
                # print("Dataset: Mesh " + mesh_folder.mesh_name + " does not have enough points " + str(len(mesh_labels[0].vertices)) + " < " + str(num_points) + ". Skipping")
                continue

            for mesh_label in mesh_labels:
                mesh_data = mesh_label.convert_to_data(outputs_at, desired_label_names,
                                                       extra_vertex_label_names=extra_vertex_label_names,
                                                       extra_global_label_names=extra_global_label_names)



                if not np.isfinite(mesh_data.vertices).all() or not np.isfinite(mesh_data.faces).all():
                    print("Dataset: Nan found in mesh. skipping", mesh_folder.mesh_name, "Recommending manual deletion")
                    continue

                if not np.isfinite(mesh_data.labels).all():
                    print("Dataset: nan in labels. skipping", mesh_folder.mesh_name, "Recommending manual deletion")
                    continue

                # Get a random permutation
                permutation = util.get_permutation_for_list(mesh_data.vertices, num_points)
                mesh_data.vertices = mesh_data.vertices[permutation]
                mesh_data.augmented_vertices = mesh_data.augmented_vertices[permutation]
                mesh_data.labels = mesh_data.labels[permutation]
                # Remove the faces
                mesh_data.faces = None

                cloud_data_list.append(mesh_data)

        return cloud_data_list

    def load_numpy_pointcloud(self, num_clouds, num_points, augmentations: List[Augmentation] | str, outputs_at="global",
                               desired_label_names=None, extra_vertex_label_names=None, extra_global_label_names=None):
        cloud_data_list = self.load_pointcloud_data(num_clouds, num_points, augmentations, outputs_at,
                               desired_label_names, extra_vertex_label_names, extra_global_label_names)
        clouds = []
        augmented_clouds = []
        labels = []
        for cloud_data in cloud_data_list:
            clouds.append(cloud_data.vertices)
            augmented_clouds.append(cloud_data.augmented_vertices)
            labels.append(cloud_data.labels)

        return np.stack(clouds), np.stack(augmented_clouds), np.stack(labels)

    def load_numpy_meshes(self, num_meshes, augmentations: List[Augmentation], outputs_at="global",
                          desired_label_names=None, extra_vertex_label_names=None, extra_global_label_names=None):
        mesh_data_list = self.load_mesh_data(num_meshes, augmentations, outputs_at,
                                             desired_label_names, extra_vertex_label_names, extra_global_label_names)
        vertices = []
        augmented_vertices = []
        faces = []
        labels = []
        for mesh_data in mesh_data_list:
            vertices.append(mesh_data.vertices)
            augmented_vertices.append(mesh_data.augmented_vertices)
            faces.append(mesh_data.faces)
            labels.append(mesh_data.labels)

        return vertices, augmented_vertices, faces, labels



#### RUN ############################

parser = argparse.ArgumentParser()

parser.add_argument('--task_id', default=0, required=False)
parser.add_argument('--num_tasks', default=1, required=False)
parser.add_argument('--input_folder', required=False)
parser.add_argument('--output_folder', required=False)

if __name__=="__main__":
    args = parser.parse_args()
    my_task_id = int(args.task_id)
    num_tasks = int(args.num_tasks)

    if args.input_folder is not None:
        base_folder = paths.RAW_DATASETS_PATH + args.input_folder
    else:
        base_folder = paths.RAW_DATASETS_PATH + "UnitTest/train/"
    if args.output_folder is not None:
        desired_path = paths.CACHED_DATASETS_PATH + args.output_folder
    else:
        desired_path = paths.CACHED_DATASETS_PATH + "unit_test/train/"

    only_points = False
    num_points = 8192
    generate_intial_folders = True
    max_submesh_index = 10

    if generate_intial_folders:
        start_from = 0
        # First generate initial folders and add initial pose
        Path(desired_path).mkdir(parents=True, exist_ok=True)

        # Grab all meshes
        origin_dataset_manager = FolderManager.DirectoryPathManager(base_path=base_folder, base_unit_is_file=True)
        file_paths = origin_dataset_manager.file_paths[start_from:]
        num_files = len(file_paths)

        for i in tqdm(range(my_task_id, num_files, num_tasks)):
            file_path = file_paths[i]
            submesh_index = get_submesh_index(file_path.file_name)
            if submesh_index > max_submesh_index:
                continue
            # Check validity
            default_augmentation = Augmentation(scale=np.array([1.0, 1.0, 1.0]),
                                                rotation=np.array([0.0, 0.0, 0.0]))
            mesh = trimesh.load(file_path.as_absolute())
            mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
            # if (not only_points) and (not mesh_aux.vertex_normals_valid()):
            #     print("Vertex Normals invalid")
            #     continue

            mesh_labels = calculate_mesh_labels(mesh=mesh, augmentation=default_augmentation,
                                                only_points=only_points, num_points=num_points)
            if mesh_labels is None:
                continue

            # save
            mesh_folder = MeshFolder(dataset_path=desired_path, mesh_name=file_path.as_subfolder_string(),
                                     only_points=only_points, num_points=num_points)
            Path(mesh_folder.mesh_dir_path).mkdir(parents=True, exist_ok=True)
            mesh_folder.initialize_folder(mesh)
            mesh_folder.calculate_and_save_augmentation(default_augmentation, override=True)

        # Now create the splits file
        create_splits_file(desired_path)

    # Now add augmentations
    augmentation_list = []
    #
    # augmentation_list.append(Augmentation(scale=np.array([1.0, 1.0, 1.0]),
    #                                       rotation=np.array([0.0, 0.0, 0.0])))
    # ========= Orientations ===========
    # augmentation_list.append(Augmentation(scale=np.array([1.0, 1.0, 1.0]),
    #                                       rotation=np.array([np.pi, 0.0, 0.0])))
    # augmentation_list.append(Augmentation(scale=np.array([1.0, 1.0, 1.0]),
    #                                       rotation=np.array([np.pi / 2, 0.0, 0.0])))
    # augmentation_list.append(Augmentation(scale=np.array([1.0, 1.0, 1.0]),
    #                                       rotation=np.array([-np.pi / 2, 0.0, 0.0])))
    # # ========= Scalings ============
    augmentation_list.append(Augmentation(scale=np.array([2.0, 1.0, 1.0]),
                                          rotation=np.array([0.0, 0.0, 0.0])))
    augmentation_list.append(Augmentation(scale=np.array([1.0, 2.0, 1.0]),
                                          rotation=np.array([0.0, 0.0, 0.0])))
    augmentation_list.append(Augmentation(scale=np.array([1.0, 1.0, 2.0]),
                                          rotation=np.array([0.0, 0.0, 0.0])))

    mesh_folder_paths = os.listdir(desired_path)
    num_files = len(mesh_folder_paths)
    for i in tqdm(range(my_task_id, num_files, num_tasks)):
        mesh_folder_path = mesh_folder_paths[i]
        mesh_folder = MeshFolder(desired_path, mesh_folder_path,
                                 only_points=only_points, num_points=num_points)
        for augmentation in augmentation_list:
            mesh_folder.calculate_and_save_augmentation(augmentation, override=False)








