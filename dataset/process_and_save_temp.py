import trimesh_util
import trimesh
import numpy as np
from pathlib import Path
import json
from typing import List
import os

class Augmentation:
    pass

class MeshLabels:
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
        return self.vertex_labels[:, :, label_indices]

    def get_global_labels(self, desired_global_label_names):
        label_indices = []
        for desired_label in desired_global_label_names:
            label_indices.append(self.global_label_names.index(desired_label))
        return self.global_labels[:, :, label_indices]

    def convert_to_data(self, outputs_at, label_names,
                        extra_vertex_label_names=None, extra_global_label_names=None):
        extra_vertex_labels = self.get_vertex_labels(extra_vertex_label_names)
        extra_global_labels = self.get_global_labels(extra_global_label_names)
        vertices = self.vertices
        augmented_vertices = np.stack(vertices, extra_vertex_labels, extra_global_labels) # TODO
        if outputs_at == "vertices":
            labels = self.get_vertex_labels(label_names)
        else:
            labels = self.get_global_labels(label_names)
        return MeshData(vertices=vertices, augmented_vertices=augmented_vertices, faces=self.faces, labels=labels)
        # return vertices, self.faces, augmented_vertices, labels

def create_mesh_labels(mesh, augmentation: Augmentation=None) -> MeshLabels:
    if augmentation is not None:
        transforms = augmentation.get_transforms()
        mesh = trimesh_util.get_transformed_mesh_trs(mesh,
                                                     scale=transforms.scale,
                                                     orientation=transforms.orientation)
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

    _, gaps, num_samples = mesh_aux.calculate_gaps_at_points(points=vertices,
                                                             normals=normals,
                                                             return_num_samples=True)
    if num_samples != num_vertices:
        print("ERROR not all vertices returned gaps")

    _, curvatures, num_samples = mesh_aux.calculate_curvature_at_points(origins=vertices,
                                                                        face_ids=None,
                                                                        curvature_method="abs",
                                                                        use_abs=False,
                                                                        return_num_samples=True)
    if num_samples != num_vertices:
        print("ERROR not all vertices returned curvature")

    surface_area = mesh_aux.surface_area
    volume = mesh_aux.volume
    centroid = np.mean(vertices, axis=0)

    vertex_labels = [thicknesses, gaps, normals[:, 0], normals[:, 1], normals[:, 2], curvatures]
    global_labels = [surface_area, volume, centroid[0], centroid[1], centroid[2]]

    vertex_labels = np.stack(vertex_labels)
    global_labels = np.array([global_labels])

    return MeshLabels(vertices=vertices, faces=faces, vertex_labels=vertex_labels,
                      global_labels=global_labels, vertex_label_names=vertex_label_names,
                      global_label_names=global_label_names)

class MeshFolder:
    def __init__(self, dataset_path, mesh_name):
        self.dataset_path = dataset_path
        self.mesh_name = mesh_name
        self.mesh_path = dataset_path + mesh_name + "/"
        self.mesh_file_path = self.mesh_path + "mesh.stl"

    def initialize_folder(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.mesh.export(self.mesh_path + "mesh.stl")
        # self.num_vertices = len(self.mesh.vertices)
        # self.num_faces = len(self.mesh.faces)

    # def load_from_folder(self):
    #     # grab mesh
    #     # grab manifest
    #     self.mesh = trimesh.load(self.mesh_path + "mesh.stl")
    #     self.num_vertices = len(self.mesh.vertices)
    #     self.num_faces = len(self.mesh.faces)

    def create_new_augmentation(self, augmentation: Augmentation, override):
        # First check if augmentation already exists
        if self.augmentation_is_cached(augmentation):
            if override:
                os.delete_file(augmentation.name)
            else:
                pass

        mesh = trimesh.load(self.mesh_path + "mesh.stl")

        # do the save
        mesh_labels = create_mesh_labels(mesh=mesh, augmentation=augmentation)

        # Save
        Path(self.mesh_path + augmentation.as_string()).mkdir(parents=True, exist_ok=True)
        # save vertex
        # save global
        # save manifest: label names, global names, augmentation values
        # save vertices
        # save faces

    def augmentation_is_cached(self, augmentation: Augmentation):
        directories = DirectoryManager.get_files(self.mesh_path)
        return augmentation.as_string() in directories

    def load_mesh_with_augmentation(self, augmentation: Augmentation):
        aug_path = self.mesh_path + augmentation.as_string() + "/"
        manifest = json.loads(aug_path + "manifest.json")
        vertex_label_names = manifest["vertex_label_names"]
        global_label_names = manifest["global_label_names"]
        vertices = np.load(aug_path + "vertices.npy")
        faces = np.load(aug_path + "faces.npy")
        vertex_labels = np.load(aug_path + "vertex_labels.npy")
        global_labels = np.load(aug_path + "global_labels.npy")

        return MeshLabels(vertices=vertices, faces=faces, vertex_labels=vertex_labels,
                          global_labels=global_labels, vertex_label_names=vertex_label_names,
                          global_label_names=global_label_names)

class MeshData:
    # This is the data that gets loaded
    def __init__(self, vertices, faces, augmented_vertices, labels):
        self.vertices = vertices
        # Inputs
        self.faces = faces
        self.augmented_vertices = augmented_vertices
        # Outputs
        self.labels = labels

class PointCloudData:
    # This is the data that gets loaded
    def __init__(self, vertices, augmented_vertices, labels):
        self.vertices
        # Inputs
        self.augmented_vertices
        # Outputs
        self.labels
        self.labels_at

class DatasetManager:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # get mesh folders
        self.mesh_folders = os.list_dir(self.dataset_path)
        self.mesh_folders.sort()


    def get_meshes(self, num_meshes, augmentations: List[Augmentation]):
        # grab num_meshes random meshes
        mesh_folders = self.mesh_folders[:num_meshes]
        mesh_labels = []
        for mesh_folder in mesh_folders:
            mesh_folder = MeshFolder(self.dataset_path, mesh_folder)
            for augmentation in augmentations:
                mesh_label = mesh_folder.load_mesh_with_augmentation(augmentation)
                mesh_labels.append(mesh_label)

        return mesh_labels

    def get_point_clouds(self, num_point_clouds, num_points, augmentations: List[Augmentation]):
        pass




