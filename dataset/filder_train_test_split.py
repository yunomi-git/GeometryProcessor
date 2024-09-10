# This takes an existing dataset of stls (or other), splits into individual pieces, then centers/normalizes them.

import trimesh
import trimesh_util
import paths
from pathlib import Path
import os
from tqdm import tqdm
import FolderManager
import shutil

# if __name__=="__main__":
#     original_path = paths.HOME_PATH + "../Datasets/" + "Thingi10k_Remesh_Normalized/"
#     new_path = paths.HOME_PATH + "../Datasets/" + "Thingi10k_Ready/"
#     Path(new_path).mkdir(exist_ok=True)
#     Path(new_path + "train").mkdir(exist_ok=True)
#     Path(new_path + "test").mkdir(exist_ok=True)
#
#     ### Single Folder
#     folder_manager = FolderManager.DirectoryPathManager(original_path, max_files_per_subfolder=-1, base_unit_is_file=True)
#
#     files = folder_manager.file_paths
#     num_files = len(files)
#
#     # perform split
#     num_train = int(0.8 * num_files)
#     train_files = files[:num_train]
#     test_files = files[num_train:]
#
#     for file_path in tqdm(train_files):
#         shutil.copyfile(original_path + file_path.as_relative(extension=True),
#                         new_path + "train/" + file_path.as_relative(extension=True))
#
#     for file_path in tqdm(test_files):
#         shutil.copyfile(original_path + file_path.as_relative(extension=True),
#                         new_path + "test/" + file_path.as_relative(extension=True))




    # for file_path in tqdm(folder_manager.file_paths):
    #     try:
    #         mesh = trimesh.load(file_path.as_absolute())
    #     except:
    #         continue
    #     mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    #     if not mesh_aux.is_valid:
    #         continue
    #
    #     sub_meshes = trimesh_util.get_valid_submeshes(mesh)
    #     i_submesh = 0
    #     for submesh in sub_meshes:
    #         submesh = trimesh_util.normalize_mesh(submesh, center=True, normalize_scale=True)
    #         Path(new_path + file_path.subfolder_path).mkdir(parents=True, exist_ok=True)
    #         submesh.export(new_path + file_path.as_relative(extension=False) + "_" + str(i_submesh) + ".stl")
    #         i_submesh += 1

