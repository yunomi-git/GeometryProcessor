import trimesh
import trimesh_util
from trimesh_util import MeshAuxilliaryInfo
import paths
import random
import numpy as np
import util
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
import math
from dataset.process_and_save import MeshDatasetFileManager, get_augmented_mesh



if __name__ == "__main__":
    metrics = util.DictionaryList()


    data_root_dir = paths.HOME_PATH + "data_augmentations/"
    data_manager = MeshDatasetFileManager(data_root_dir)
    data_files =  data_manager.get_target_files(absolute=True)
    for data_file in tqdm(data_files):
        # print(data_file)
        # Load the file
        file_path = data_file
        with open(file_path, 'r') as f:
            mesh_data_master = json.load(f)
        for mesh_data in mesh_data_master["instances"]:
            if mesh_data["scale"] > 1000:
                continue
            # if mesh_data["vertices"] > 1e6:
            #     continue
            # if mesh_data["vertices"] < 1e2:
            #     continue
            # if (mesh_data["thickness_violation"] > 0.5):
            #     mesh = trimesh.load(data_root_dir + mesh_data_master["mesh_relative_path"])
            #     mesh = get_augmented_mesh(mesh, mesh_data)
            #     mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
            #     points, values = mesh_aux.calculate_thicknesses_samples()
            #     trimesh_util.show_sampled_values(mesh, points=points, values=values)
            # mesh = trimesh.load(data_root_dir + mesh_data_master["mesh_relative_path"])
            # mesh = get_augmented_mesh(mesh, mesh_data)
            # mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
            # points, values = mesh_aux.calculate_overhangs_samples()
            # trimesh_util.show_sampled_values(mesh, points=points, values=values)
            if math.isnan(mesh_data["thickness_violation"]):
                continue
            metrics.add_element(mesh_data)

    data = pd.DataFrame(data=metrics.master_list)

    sns.histplot(data=data, x="vertices", log_scale=True)
    plt.show()
    sns.histplot(data=data, x="scale", log_scale=True)
    plt.show()
    sns.histplot(data=data, x="volume", log_scale=True)
    plt.show()
    sns.histplot(data=data, x="overhang_violation", log_scale=(False, True))
    plt.show()
    sns.histplot(data=data, x="stairstep_violation", log_scale=(False, True))
    plt.show()
    sns.histplot(data=data, x="thickness_violation", log_scale=(False, True))
    plt.show()
    sns.histplot(data=data, x="gap_violation", log_scale=(False, True))
    plt.show()

    # sub_df = data[["vertices", "scale", "volume", "overhang_violation", "stairstep_violation", "thickness_violation", "gap_violation"]]
    # sns.pairplot(sub_df)
    # plt.show()