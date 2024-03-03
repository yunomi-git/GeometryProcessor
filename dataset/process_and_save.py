import trimesh
import trimesh_util
from trimesh_util import MeshAuxilliaryInfo
import paths
import numpy as np
from heuristic_prediction import printability_metrics
from tqdm import tqdm
import json
from dataset.remeshing import fix_mesh, remesh_and_save
import pymesh
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R

class MeshDatasetFileManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def get_mesh_path(self, absolute=True):
        if absolute:
            return self.root_dir + "mesh/"
        else:
            return "mesh/"

    def get_target_path(self, absolute=True):
        if absolute:
            return self.root_dir + "target/"
        else:
            return "target/"

    def get_target_files(self, absolute=True):
        target_directory = self.get_target_path(absolute=True)
        if absolute:
            data_files = [(target_directory + file)
                          for file in os.listdir(target_directory)
                          if os.path.isfile(os.path.join(target_directory, file))]
        else:
            data_files = (file
                          for file in os.listdir(target_directory)
                          if os.path.isfile(os.path.join(target_directory, file)))
        return data_files

    def get_mesh_files(self, absolute=True):
        target_directory = self.get_mesh_path(absolute=True)
        if absolute:
            data_files = [(target_directory + file)
                          for file in os.listdir(target_directory)
                          if os.path.isfile(os.path.join(target_directory, file))]
        else:
            data_files = (file
                          for file in os.listdir(target_directory)
                          if os.path.isfile(os.path.join(target_directory, file)))
        return data_files
    def load_base_mesh_from_target(self, target_path_absolute):
        with open(target_path_absolute, 'r') as f:
            target = json.load(f)
        mesh_path = self.root_dir + target["mesh_relative_path"]
        return trimesh.load(mesh_path, force="mesh")


def get_augmented_mesh(mesh: trimesh.Trimesh, augmentations):
    ###
    # Assumes that the mesh only contains 1 body
    # Augmentations: {
    # "euler_orientation":
    #       "x", "y", "z"
    # "scale":
    ###
    orientation = augmentations["orientation"]
    eulers = [orientation["z"], orientation["y"], orientation["x"]]
    scale = augmentations["scale"]

    transformed_mesh = trimesh_util.get_transformed_mesh(mesh, scale=scale, orientation=eulers)

    return transformed_mesh


def calculate_instance_target(mesh: trimesh.Trimesh, augmentations: dict):
    # Augmentations are just saved to the target. Not used for calculations
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    thickness_printability = printability_metrics.get_thickness_printability(mesh_aux)
    gap_printability = printability_metrics.get_gap_printability(mesh_aux)
    instance_target = {
        "vertices": mesh_aux.num_vertices,
        "edges": mesh_aux.num_edges,
        "faces": mesh_aux.num_facets,
        "bound_length": np.max(mesh_aux.bound_length),
        "is_manifold": mesh.is_watertight,
        "volume": mesh.volume,
        "overhang_violation": printability_metrics.get_overhang_printability(mesh_aux)[2],
        "stairstep_violation": printability_metrics.get_stairstep_printability(mesh_aux)[2],
        "thickness_violation": thickness_printability[2],
        "min_thickness": thickness_printability[3],
        "gap_violation": gap_printability[2],
        "min_gap": gap_printability[3],
    }
    instance_target.update(augmentations)
    return instance_target

def save_base_mesh_and_target(mesh, mesh_file_manager: MeshDatasetFileManager, mesh_name):
    # Only take the largest body in a mesh
    if mesh.body_count > 1:
        splits = list(mesh.split(only_watertight=False))
        largest_volume = 0
        largest_submesh = None
        for submesh in splits:
            temp_volume = submesh.volume
            if temp_volume > largest_volume:
                largest_volume = temp_volume
                largest_submesh = submesh
        mesh = largest_submesh

    target = {
        "base_name": mesh_name,
        "mesh_relative_path": mesh_file_manager.get_mesh_path(absolute=False) + mesh_name + ".stl",
        "instances": []
    }

    mesh.export(mesh_file_manager.get_mesh_path(absolute=True) + mesh_name + ".stl", file_type="stl")
    target_path_absolute = mesh_file_manager.get_target_path(absolute=True) + mesh_name + ".json"
    with open(target_path_absolute, 'w') as f:
        json.dump(target, f)
        
    # Add the first instance
    augmentations = {
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        },
        "scale": 1
    }
    calculate_and_append_augmentation(target=target,
                                      mesh=mesh,
                                      augmentations=augmentations)
    with open(target_path_absolute, 'w') as f:
        json.dump(target, f)
    return target, mesh



def calculate_and_append_augmentation(mesh: trimesh.Trimesh, target: dict, augmentations: dict):
    mesh = get_augmented_mesh(mesh, augmentations)
    instance_target = calculate_instance_target(mesh, augmentations=augmentations)
    target["instances"].append(instance_target)


def generate_base_dataset(data_root_dir, source_mesh_filenames, save_prefix, mesh_scale=1.0):
    # Setup data path
    mesh_file_manager = MeshDatasetFileManager(root_dir=data_root_dir)
    Path(mesh_file_manager.get_mesh_path(absolute=True)).mkdir(parents=True, exist_ok=True)
    Path(mesh_file_manager.get_target_path(absolute=True)).mkdir(parents=True, exist_ok=True)

    i = 0
    for mesh_path in tqdm(source_mesh_filenames):
        base_name = save_prefix + "_" + "mesh" + str(i)

        mesh = trimesh.load(mesh_path, force='mesh')

        if not trimesh_util.mesh_is_valid(mesh):
            print("mesh skipped:", i)
            continue

        if not mesh.is_watertight:
            print("mesh not watertight. skipped:", i)
            continue

        mesh.apply_scale(mesh_scale)
        try:
            # mesh_info, mesh_to_save = calculate_mesh_info(mesh, save_relative_path=save_mesh_relative_path, orig_mesh_path=mesh_path)
            save_base_mesh_and_target(mesh, mesh_file_manager=mesh_file_manager, mesh_name=base_name)
            i += 1
        except:
            continue

def generate_augmentations_for_dataset(mesh_file_manager: MeshDatasetFileManager, augmentation_list):
    # Setup data path
    target_path_absolute_list = mesh_file_manager.get_target_files(absolute=True)

    for target_path in tqdm(target_path_absolute_list):
        with open(target_path, "r")as f:
            target = json.load(f)

        mesh = mesh_file_manager.load_base_mesh_from_target(target_path_absolute=target_path)

        for augmentation in augmentation_list:
            calculate_and_append_augmentation(mesh=mesh, target=target, augmentations=augmentation)

        with open(target_path, 'w') as f:
            json.dump(target, f)

def quick_edit(data_root_directory):
    mesh_file_manager = MeshDatasetFileManager(data_root_directory)
    target_files = mesh_file_manager.get_target_files(absolute=True)
    for file in target_files:
        with open(file, 'r') as f:
            target = json.load(f)
        relative_path = target["mesh_relative_path"]
        period = relative_path.find(".")
        target["mesh_relative_path"] = relative_path[:period] + ".stl"
        with open(file, 'w') as f:
            json.dump(target, f)



if __name__ == "__main__":
    # quick_edit(paths.HOME_PATH + "data_augmentations/")
    # 1/0
    # Generation Parameters
    do_generate_base_dataset = False
    if do_generate_base_dataset:
        use_onshape = True
        source_mesh_filenames = []
        if use_onshape:
            min_range = 0
            max_range = 290
            mesh_scale = 25.4
            prefix = "onshape"
            for i in range(min_range, max_range):
                source_mesh_filenames.append(paths.get_onshape_stl_path(i, get_by_order=True))
        else:
            min_range = 0
            max_range = 2000
            mesh_scale = 1.0
            prefix = "thing"
            for i in range(min_range, max_range):
                source_mesh_filenames.append(paths.get_thingiverse_stl_path(i, get_by_order=True))
    else:
        only_add_augmentations = False
        add_basic_augmentations = True
        num_extra_augmentations = 5
    overwrite = False
    outputs_save_path = paths.HOME_PATH + "data_augmentations/"




    # Setup data path
    mesh_file_manager = MeshDatasetFileManager(root_dir=outputs_save_path)
    Path(mesh_file_manager.get_mesh_path(absolute=True)).mkdir(parents=True, exist_ok=True)
    Path(mesh_file_manager.get_target_path(absolute=True)).mkdir(parents=True, exist_ok=True)


    if do_generate_base_dataset:
        generate_base_dataset(data_root_dir=outputs_save_path,
                              source_mesh_filenames=source_mesh_filenames,
                              save_prefix=prefix,
                              mesh_scale=mesh_scale)
    else:
        augmentation_list = []
        augmentations = {
            "orientation": {
                "x": np.pi,
                "y": 0.0,
                "z": 0.0
            },
            "scale": 1
        }
        augmentation_list.append(augmentations)
        augmentations = {
            "orientation": {
                "x": np.pi/2,
                "y": 0.0,
                "z": 0.0
            },
            "scale": 1
        }
        augmentation_list.append(augmentations)
        augmentations = {
            "orientation": {
                "x": -np.pi/2,
                "y": 0.0,
                "z": 0.0
            },
            "scale": 1
        }
        augmentation_list.append(augmentations)

        generate_augmentations_for_dataset(mesh_file_manager=mesh_file_manager,
                                           augmentation_list=augmentation_list)



