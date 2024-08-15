import trimesh
import trimesh_util
from FolderManager import DirectoryPathManager, FilePath
import paths
from dataset.pymeshlab_remesh import default_remesh
from pathlib import Path
from tqdm import tqdm
from util import Stopwatch
import sys
import os

# Grab back. From Project: rsync -uPr nyu@txe1-login.mit.edu:~/Datasets/DrivAerNet/Simplified_Remesh Datasets/DrivAerNet/Simplified_Remesh/


if __name__=="__main__":
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    override = False

    root_path = paths.RAW_DATASETS_PATH + "DrivAerNet/Simplified/"
    new_save_path = paths.RAW_DATASETS_PATH + "DrivAerNet/Simplified_Remesh/"

    directory_manager = DirectoryPathManager(root_path, base_unit_is_file=True)
    # All files that are .stl and that are not wheels
    files = [file for file in directory_manager.file_paths if (file.extension == ".stl" and
                                                               (file.file_name.lower().find("wheel")) == -1)]
    # watch = Stopwatch()
    num_files = len(files)
    for i in range(my_task_id, num_files, num_tasks):
        file = files[i]
        # watch.start()
        # print("remeshing")
        new_file_directory = new_save_path + file.subfolder_path
        Path(new_file_directory).mkdir(parents=True, exist_ok=True)
        new_file_path = new_save_path + file.as_relative(extension=False) + ".stl"
        if not override and os.path.exists(new_file_path):
            print("File exists. Skipping", new_file_path)
            continue
        temp_file_path = new_save_path + file.as_relative(extension=False) + "_temp.stl"

        try:
            default_remesh(file_path=file.as_absolute(), out_path=temp_file_path)
        except Exception as e:
            print("Error remeshing:", e)
            os.rmdir(new_file_directory)
            continue

        try:
            mesh = trimesh.load(temp_file_path)
        except Exception as e:
            print("Error loading trimesh:", e)
            os.remove(temp_file_path)
            os.rmdir(new_file_directory)
            continue

        try:
            # Mirror
            mesh = trimesh_util.mirror_surface(mesh, plane="y", process=False)
            # Normalize and center
            mesh = trimesh_util.normalize_mesh(mesh, center=True, normalize_scale=True)
            mesh.export(new_file_path)
            os.remove(temp_file_path)
            print("Saved", new_file_path)
        except Exception as e:
            print("Error mirroring", e)
            os.remove(temp_file_path)
            os.rmdir(new_file_directory)
            continue

