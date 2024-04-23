# Compare the loading speed of 1 and n meshes
import trimesh
from util import Stopwatch
import paths
import process_and_save as ps
import potpourri3d as ppd

if __name__=="__main__":
    base_directory = paths.DATA_PATH + "data_mini/"
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