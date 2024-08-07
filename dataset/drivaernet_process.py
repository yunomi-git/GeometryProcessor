import trimesh
import trimesh_util
from dataset.FolderManager import DirectoryPathManager, FilePath
import paths
from dataset.pymeshlab_remesh import default_remesh
from pathlib import Path
from tqdm import tqdm
from util import Stopwatch

if __name__=="__main__":
    root_path = paths.RAW_DATASETS_PATH + "DrivAerNet/Simplified/"
    new_save_path = paths.RAW_DATASETS_PATH + "DrivAerNet/Simplified_Remesh/"
    directory_manager = DirectoryPathManager(root_path, base_unit_is_file=True)
    # All files that are .stl and that are not wheels
    files = [file for file in directory_manager.file_paths if (file.extension == ".stl" and
                                                               (file.file_name.lower().find("wheel")) == -1)]
    # watch = Stopwatch()
    for file in tqdm(files[1299:]):
        # watch.start()
        # print("remeshing")
        new_file_directory = new_save_path + file.subfolder_path
        Path(new_file_directory).mkdir(parents=True, exist_ok=True)
        new_file_path = new_save_path + file.as_relative(extension=False) + ".stl"

        default_remesh(file_path=file.as_absolute(), out_path="temp.stl")
        # watch.print_time("remesh")

        # print("Mirroring")
        try:
            mesh = trimesh.load("temp.stl")
        except Exception as e:
            print("Error loading trimesh:", e)
        # Mirror
        mesh = trimesh_util.mirror_surface(mesh, plane="y", process=False)
        # Normalize and center
        mesh = trimesh_util.normalize_mesh(mesh, center=True, normalize_scale=True)
        mesh.export(new_file_path)
        # watch.print_time("mirror")
        # Remesh

    # mirror
    # Normalize and center
    # Remesh