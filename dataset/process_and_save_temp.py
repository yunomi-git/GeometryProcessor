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

def get_submesh_index(mesh_name):
    index = mesh_name.find("_")
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
            label_indices.append(self.vertex_label_names.index(desired_label))
        return self.vertex_labels[:, label_indices]

    def get_global_labels(self, desired_global_label_names):
        label_indices = []
        for desired_label in desired_global_label_names:
            label_indices.append(self.global_label_names.index(desired_label))
        return self.global_labels[label_indices]

    def convert_to_data(self, outputs_at, label_names,
                        extra_vertex_label_names=None, extra_global_label_names=None):
        augmented_vertices = self.vertices
        if len(extra_vertex_label_names) > 0:
            extra_vertex_labels = self.get_vertex_labels(extra_vertex_label_names)
            augmented_vertices = np.stack(augmented_vertices, extra_vertex_labels)
        if len(extra_global_label_names) > 0:
            extra_global_labels = self.get_global_labels(extra_global_label_names)
            augmented_vertices = np.stack(augmented_vertices,
                                          np.repeat(extra_global_labels[np.newaxis, :], len(augmented_vertices)))


        # augmented_vertices = np.stack(vertices, extra_vertex_labels, np.repeat(extra_global_labels[np.newaxis, :], len(vertices))) # TODO
        if outputs_at == "vertices":
            labels = self.get_vertex_labels(label_names)
        else:
            labels = self.get_global_labels(label_names)
        return MeshReadyData(vertices=self.vertices, augmented_vertices=augmented_vertices,
                             faces=self.faces, labels=labels)


def calculate_mesh_labels(mesh, augmentation: Augmentation=None) -> MeshRawLabels:
    if augmentation is not None:
        mesh = trimesh_util.get_transformed_mesh_trs(mesh,
                                                     scale=augmentation.scale,
                                                     orientation=augmentation.rotation)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    vertices = mesh_aux.vertices
    faces = mesh_aux.faces
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

    normals = mesh_aux.vertex_normals
    _, thicknesses, num_samples = mesh_aux.calculate_thickness_at_points(points=vertices,
                                                                         normals=normals,
                                                                         return_num_samples=True)
    if num_samples != num_vertices:
        print("ERROR not all vertices returned thicknesses")
        return

    _, gaps, num_samples = mesh_aux.calculate_gaps_at_points(points=vertices,
                                                             normals=normals,
                                                             return_num_samples=True)
    if num_samples != num_vertices:
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
    def __init__(self, dataset_path, mesh_name):
        self.dataset_path = dataset_path
        self.mesh_name = mesh_name
        self.mesh_dir_path = dataset_path + str(mesh_name) + "/"
        self.mesh_stl_path = self.mesh_dir_path + "mesh.stl"

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
        mesh_labels = calculate_mesh_labels(mesh=mesh, augmentation=augmentation)

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
            manifest = json.load(f)
        vertex_label_names = manifest["vertex_label_names"]
        global_label_names = manifest["global_label_names"]
        vertices = np.load(aug_path + "vertices.npy")
        faces = np.load(aug_path + "faces.npy")
        vertex_labels = np.load(aug_path + "vertex_labels.npy")
        global_labels = np.load(aug_path + "global_labels.npy")

        # TODO: Meshes were saved incorrectly the first time. need to redo
        return MeshRawLabels(vertices=vertices, faces=faces, vertex_labels=vertex_labels.T,
                             global_labels=global_labels[0], vertex_label_names=vertex_label_names,
                             global_label_names=global_label_names)

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

    def load_specific_augmentations_if_available(self, augmentations: List[Augmentation] | List[str]):
        mesh_labels = []
        for augmentation in augmentations:
            if self.augmentation_is_cached(augmentation):
                mesh_labels.append(self.load_mesh_with_augmentation(augmentation))
        return mesh_labels


class MeshReadyData:
    # This is the data that gets loaded
    def __init__(self, vertices, faces, augmented_vertices, labels):
        self.vertices = vertices
        # Inputs
        self.faces = faces
        self.augmented_vertices = augmented_vertices
        # Outputs
        self.labels = labels

# class PointCloudData:
#     # This is the data that gets loaded
#     def __init__(self, vertices, augmented_vertices, labels):
#         self.vertices
#         # Inputs
#         self.augmented_vertices
#         # Outputs
#         self.labels
#         self.labels_at

class DatasetManager:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # get mesh folders
        # os.listdir(self.dataset_path)
        self.mesh_folder_names = os.listdir(self.dataset_path)
        self.mesh_folder_names.sort()

    def get_mesh_folders(self, num_meshes): #, num_meshes, augmentations: List[Augmentation]):
        # grab num_meshes random meshes
        mesh_folders = []
        mesh_names = util.get_random_n_in_list(self.mesh_folder_names, num_meshes)
        for mesh_name in mesh_names:
            mesh_folder = MeshFolder(self.dataset_path, mesh_name)
            mesh_folders.append(mesh_folder)

        return mesh_folders

    def get_point_clouds(self, num_point_clouds, num_points, augmentations: List[Augmentation]):
        pass


if __name__=="__main__":
    base_folder = paths.DATASETS_PATH + "UnitTest/train/"
    desired_path = paths.DATA_PATH + "unit_test/train/"
    # desired_path = os.path.join(paths.DATA_PATH)

    generate_intial_folders = True
    max_submesh_index = 4

    if generate_intial_folders:
        start_from = 0
        # First generate initial folders and add initial pose
        Path(desired_path).mkdir(parents=True, exist_ok=True)

        # Grab all meshes
        origin_dataset_manager = FolderManager.DirectoryPathManager(base_path=base_folder, base_unit_is_file=True)

        for file_path in tqdm(origin_dataset_manager.file_paths[start_from:]):
            submesh_index = get_submesh_index(file_path.file_name)
            if submesh_index > max_submesh_index:
                continue
            # Check validity
            default_augmentation = Augmentation(scale=np.array([1.0, 1.0, 1.0]),
                                                rotation=np.array([0.0, 0.0, 0.0]))
            mesh = trimesh.load(file_path.as_absolute())
            mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
            if not mesh_aux.vertex_normals_valid():
                print("Vertex Normals invalid")
                continue

            mesh_labels = calculate_mesh_labels(mesh=mesh, augmentation=default_augmentation)
            if mesh_labels is None:
                continue

            # save
            mesh_folder = MeshFolder(dataset_path=desired_path, mesh_name=file_path.as_subfolder_string())
            Path(mesh_folder.mesh_dir_path).mkdir(parents=True, exist_ok=True)
            mesh_folder.initialize_folder(mesh)
            mesh_folder.calculate_and_save_augmentation(default_augmentation, override=True)

    # Now add augmentations
    augmentation_list = []
    #
    # augmentation_list.append(Augmentation(scale=np.array([1.0, 1.0, 1.0]),
    #                                       rotation=np.array([0.0, 0.0, 0.0])))
    # ========= Orientations ===========
    augmentation_list.append(Augmentation(scale=np.array([1.0, 1.0, 1.0]),
                                          rotation=np.array([np.pi, 0.0, 0.0])))
    augmentation_list.append(Augmentation(scale=np.array([1.0, 1.0, 1.0]),
                                          rotation=np.array([np.pi / 2, 0.0, 0.0])))
    augmentation_list.append(Augmentation(scale=np.array([1.0, 1.0, 1.0]),
                                          rotation=np.array([-np.pi / 2, 0.0, 0.0])))
    # # ========= Scalings ============
    augmentation_list.append(Augmentation(scale=np.array([2.0, 1.0, 1.0]),
                                          rotation=np.array([0.0, 0.0, 0.0])))
    augmentation_list.append(Augmentation(scale=np.array([1.0, 2.0, 1.0]),
                                          rotation=np.array([0.0, 0.0, 0.0])))
    augmentation_list.append(Augmentation(scale=np.array([1.0, 1.0, 2.0]),
                                          rotation=np.array([0.0, 0.0, 0.0])))

    mesh_folder_paths = os.listdir(desired_path)
    for mesh_folder_path in tqdm(mesh_folder_paths):
        mesh_folder = MeshFolder(desired_path, mesh_folder_path)
        for augmentation in augmentation_list:
            mesh_folder.calculate_and_save_augmentation(augmentation, override=False)








