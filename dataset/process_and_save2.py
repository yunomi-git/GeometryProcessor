import trimesh
import trimesh_util
import paths
import numpy as np

import util
from heuristic_prediction import printability_metrics
from tqdm import tqdm
import json
import os
from pathlib import Path

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
                "s" + simplify_number(self.scale[0]) +
                simplify_number(self.scale[1]) +
                simplify_number(self.scale[2]))

class FileManager:
    def __init__()
        self.num_training

        self.num_test


    def augmentation_exists(self, mesh_name, augmentation):


    def get_mesh_name_lists(self):


    def load_meshes(augmentations_list, num_meshes, filter, labels):
        num_meshes_loaded = 0
        vertices
        faces
        vlabels
        glabels

        # First get list of all meshes
        meshes_list = get_mesh_name_lists
        for mesh_name in tqdm(meshes_list):
            MeshFolder(mesh_name)
            if filter(MeshFolder) == False:
                break

            num_meshes_loaded += 1
            for augmentation in augmentation_list
                if augmentation_exists():
                    _ = mesh_folder.load_augmentation
                    xxx.append(xxx)

            return _


class MeshFolder:
    def __init__(self, path, name):
        self.name = None

        self.path = None

        self.global_manifest = None

        self.stl_path = None

        self.vertices_name = None

        self.faces_name = None

        self.vlabel_name_name = None

        self.glabel = None



    def save_base_folder(self, mesh_name, mesh):
        manifest = {
            "num_vertices": mesh.vertices,
            "num_faces": mesh.faces,
        }
        json.dumps(manifest, self.path + "manifest.json")

    def get_original_stl(self):
        trimesh.load(self.stl_path)


    def load_augmentation(self, augmentation, vertex_label_names, global_label_names,
                                              extra_vertex_label_names=None, extra_global_label_names=None):
        aug_dir_path = self.path + augmentation_name + "/"

        manifest = json.loads(aug_dir_path + "manifest.json")
        v = np.load(aug_dir_path + "vertices.npy", allow_pickle=True)
        f = np.load(aug_dir_path + "faces.npy", allow_pickle=True)
        all_vertex_labels = np.load(aug_dir_path + "vertex_labels.npy", allow_pickle=True)
        all_global_labels = np.load(aug_dir_path + "global_labels.npy", allow_pickle=True)
        all_vertex_label_names = manifest["vertex_label_names"]
        all_global_label_names = manifest["global_label_names"]

        # grab relevant labels
        vertex_labels = all_vertex_labels[:, :, util.get_indices_a_in_b(vertex_label_names,
                                                                        all_vertex_label_names)]
        extra_vertex_labels = all_vertex_labels[:, :, util.get_indices_a_in_b(extra_vertex_label_names,
                                                                              all_vertex_label_names)]
        global_labels = all_global_labels[:, util.get_indices_a_in_b(global_label_names,
                                                                     all_global_label_names)]
        extra_global_labels = all_global_labels[:, util.get_indices_a_in_b(extra_global_label_names,
                                                                           all_global_label_names)]

        return v, f, vertex_labels, global_labels, extra_vertex_labels, extra_global_labels

    def save_augmentation(self, mesh, augmentation: Augmentation, vertex_labels, vertex_label_names, global_labels, global_label_names):
        aug_dir_path = self.path + augmentation.as_string() + "/"
        manifest = {
            "augmentation": augmentation.as_json(),
            "vertex_label_names": vertex_label_names,
            "global_label_names": global_label_names
        }
        json.dump(manifest, aug_dir_path + "manifest.json")
        Path(aug_dir_path).mkdir(parents=True, exist_ok=False)
        np.save(aug_dir_path + "vertices", mesh.vertices)
        np.save(aug_dir_path + "faces", mesh.faces)
        np.save(aug_dir_path + "vertex_labels", vertex_labels)
        np.save(aug_dir_path + "global_labels", global_labels)

        return
