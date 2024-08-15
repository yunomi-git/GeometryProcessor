import paths
import torch
import numpy as np
from dataset.process_and_save_temp import DatasetManager
import diffusion_net
from shape_regression.diffusionnet_model import mesh_is_valid_diffusion
import argparse

# Grab back. From Project: rsync -uPr nyu@txe1-login.mit.edu:~/Datasets/DrivAerNet/Simplified_Remesh Datasets/DrivAerNet/Simplified_Remesh/

parser = argparse.ArgumentParser()

parser.add_argument('--task_id', default=0, required=False)
parser.add_argument('--num_tasks', default=1, required=False)
parser.add_argument('--input_folder', required=True)

if __name__=="__main__":
    args = parser.parse_args()
    my_task_id = int(args.task_id)
    num_tasks = int(args.num_tasks)

    root_path = paths.CACHED_DATASETS_PATH + args.input_folder #"DrivAerNet/train/"
    op_cache_dir = root_path + "../op_cache/"

    augmentations = "all"
    k_eig = 64

    # Open the base directory and get the contents
    file_manager = DatasetManager(root_path)
    num_files = len(file_manager.mesh_folder_names)
    mesh_folders = file_manager.get_mesh_folders(num_files)

    print("loading augmentations: ")
    print(augmentations)

    # watch = Stopwatch()
    num_files = len(mesh_folders)
    for i in range(my_task_id, num_files, num_tasks):
        mesh_folder = mesh_folders[i]
        mesh_labels = mesh_folder.load_mesh_with_augmentations(augmentations)

        for mesh_label in mesh_labels:
            mesh_data = mesh_label.convert_to_data("vertices", label_names=[],
                                                   extra_vertex_label_names=[],
                                                   extra_global_label_names=[])

            verts = mesh_data.vertices
            aug_verts = mesh_data.augmented_vertices
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
                print("Dataset: Face index exceeds vertices. skipping", mesh_folder.mesh_name,
                      "Recommending manual deletion")
                continue

            # Attempt to get eigen decomposition. If cannot, skip
            try:
                diffusion_net.geometry.get_operators(tensor_vert, tensor_face, k_eig=k_eig,
                                                     op_cache_dir=op_cache_dir)
            except:  # Or valueerror or ArpackError
                print("Dataset: Error calculating decomposition. Skipping", mesh_folder.mesh_name,
                      "Recommending manual deletion")
                continue

