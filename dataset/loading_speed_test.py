# Compare the loading speed of 1 and n meshes
import numpy as np
import trimesh
from util import Stopwatch
import paths
import process_and_save as ps
import potpourri3d as ppd
import os
from pathlib import Path
import FolderManager as folders

def run_1():
    base_directory = paths.CACHED_DATASETS_PATH + "data_mini/"
    file_manager = ps.MeshDatasetFileManager(base_directory)
    files = file_manager.get_mesh_files()

    # files = files[:300]
    stopwatch = Stopwatch()
    stopwatch.start()
    for file in files:
        trimesh.load(file)
    time = stopwatch.get_elapsed_time()
    print("Trimesh: ", time)

    stopwatch.start()
    for file in files:
        ppd.read_mesh(file)
    time = stopwatch.get_elapsed_time()
    print("PPD: ", time)

    stopwatch.start()
    file_manager.load_numpy_pointclouds(sampling_method="even", desired_label_names=[])
    time = stopwatch.get_elapsed_time()
    print("numpy: ", time)

def create_test_2_data():
    base_path = paths.HOME_PATH + "dataset/speed_test/"
    file_names = os.listdir(base_path + "stls/")
    file_names.sort()
    original_extension = file_names[0][file_names[0].find("."):]
    file_names = [file_name[:file_name.find(".")] for file_name in file_names]

    # save their faces and vertices
    for file in file_names:
        v, f = ppd.read_mesh(base_path + "stls/" + file + original_extension)
        subfolder_path = base_path + "numpy/" + file + "/"
        Path(subfolder_path).mkdir(exist_ok=True)
        np.save(subfolder_path + "vertices", v)
        np.save(subfolder_path + "faces", f)


def run_2():
    # This test tests load speed of 1000 meshes.
    base_path = paths.HOME_PATH + "dataset/speed_test/"
    stopwatch = Stopwatch()

    # Load and test ppd3d on stls
    file_names = os.listdir(base_path + "stls/")
    stopwatch.start()
    for file in file_names:
        v, f = ppd.read_mesh(base_path + "stls/" + file)
    time = stopwatch.get_elapsed_time()
    print("PPD: ", time)

    # Load and test numpy
    folder_names = os.listdir(base_path + "numpy/")
    stopwatch.start()
    for folder in folder_names:
        v = np.load(base_path + "numpy/" + folder + "/vertices.npy", allow_pickle=True)
        f = np.load(base_path + "numpy/" + folder + "/faces.npy", allow_pickle=True)
    time = stopwatch.get_elapsed_time()
    print("numpy: ", time)


    pass

if __name__=="__main__":
    run_2()
