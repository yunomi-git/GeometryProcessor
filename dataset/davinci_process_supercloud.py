import trimesh
import trimesh_util
from FolderManager import DirectoryPathManager, FilePath
import paths
from dataset.pymeshlab_remesh import default_remesh_with_checks
from pathlib import Path
from tqdm import tqdm
from util import Stopwatch
import sys
import os
import argparse

# Grab back. From Project: rsync -uPr nyu@txe1-login.mit.edu:~/Datasets/DrivAerNet/Simplified_Remesh Datasets/DrivAerNet/Simplified_Remesh/

# There are too many folders, so instead of doing all folders simultaneously, do them one at a time
parser = argparse.ArgumentParser()

parser.add_argument('--task_id', default=0, required=False)
parser.add_argument('--num_tasks', default=1, required=False)

if __name__=="__main__":
    args = parser.parse_args()
    my_task_id = int(args.task_id)
    num_tasks = int(args.num_tasks)

    # override = False

    davinci_log_file = "davinci_log.txt"

    master_path = paths.RAW_DATASETS_PATH + "DaVinci/stls/"
    folders = os.listdir(master_path)
    folders.sort()
    for folder in folders:
        print(folder)
        root_path = paths.RAW_DATASETS_PATH + "DaVinci/stls/" + folder + "/"
        new_save_path_combined = paths.RAW_DATASETS_PATH + "DaVinci/remesh_combined/" + folder + "/"
        new_save_path_split = paths.RAW_DATASETS_PATH + "DaVinci/remesh_split/" + folder + "/"

        directory_manager = DirectoryPathManager(root_path, base_unit_is_file=True)
        # All files that are .stl and that are not wheels
        # files = [file for file in directory_manager.file_paths if (file.extension == ".stl" and
        #                                                            (file.file_name.lower().find("wheel")) == -1)]
        print("Total num files: ", len(directory_manager.file_paths))
        files = directory_manager.file_paths
        # watch = Stopwatch()
        num_files = len(files)
        for i in range(my_task_id, num_files, num_tasks):
            file = files[i]
            with open(davinci_log_file, "w") as f:
                f.write(file.as_relative())
            # watch.start()
            # print("remeshing")
            Path(new_save_path_split + file.subfolder_path).mkdir(parents=True, exist_ok=True)
            Path(new_save_path_combined + file.subfolder_path).mkdir(parents=True, exist_ok=True)

            new_file_path_combined = new_save_path_combined + file.as_relative(extension=False) + ".stl"
            new_file_path_split = lambda x: new_save_path_split + file.as_relative(extension=False) + "_" + str(x) + ".stl"
            # if not override and os.path.exists(new_file_path):
            #     print("File exists. Skipping", new_file_path)
            #     continue
            temp_file_path = new_save_path_split + file.as_relative(extension=False) + "_temp.stl"

            # Do the remeshing in meshlab
            error_msg = default_remesh_with_checks(file_path=file.as_absolute(), out_path=temp_file_path)
            if error_msg != 0:
                print("Error remeshing ^")
                continue

            # Load to trimesh
            try:
                mesh = trimesh.load(temp_file_path)
            except Exception as e:
                print("Error loading trimesh:", e)
                os.remove(temp_file_path)
                # os.rmdir(new_file_directory)
                continue

            # Save the default
            mesh.export(new_file_path_combined)

            # Split, normalize, and center
            try:
                # Split
                sub_meshes = trimesh_util.get_valid_submeshes(mesh)
                MAX_SUBMESHES_TO_SAVE = 4
                num_submeshes_to_save = len(sub_meshes)
                if num_submeshes_to_save > MAX_SUBMESHES_TO_SAVE:
                    num_submeshes_to_save = MAX_SUBMESHES_TO_SAVE

                for i in range(num_submeshes_to_save):
                    # Normalize and center
                    submesh = trimesh_util.normalize_mesh(sub_meshes[i], center=True, normalize_scale=True)
                    submesh.export(new_file_path_split(i))
                    print("Saved", new_file_path_split(i))

                os.remove(temp_file_path)

            except Exception as e:
                print("Error splitting and normalizing:", e)
                os.remove(temp_file_path)
                # os.rmdir(new_file_directory)
                continue

