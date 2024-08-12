import pymeshlab
import paths
from tqdm import tqdm
import util as util
import trimesh
import trimesh_util as trimesh_util
import numpy as np
import os
from pathlib import Path
from dataset.FolderManager import DirectoryPathManager

time = util.Stopwatch()

def default_remesh(file_path, out_path=None, show=False):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_path)

    ms.meshing_surface_subdivision_midpoint(iterations=3)
    # ms.meshing_decimation_clustering(threshold=pymeshlab.PercentageValue(0.1))
    ms.meshing_isotropic_explicit_remeshing(iterations=5)


    if out_path is not None:
        ms.save_current_mesh(out_path, save_face_color=False)
    if show:
        ms.show_polyscope()

def default_remesh_with_checks(file_path, out_path=None):
    try:
        mesh = trimesh.load(file_path)
        mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    except:
        print("error loading trimesh", file_path)
        return -1

    if not mesh_aux.is_valid:
        print("not valid: ", file_path)
        return -1

    if not mesh.is_watertight:
        print("not watertight: ", file_path)
        return -1

    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file_path)
    except:
        print("error loading meshlab", file_path)
        return -1

    try:
        ms.meshing_surface_subdivision_midpoint(iterations=3)
        ms.meshing_isotropic_explicit_remeshing(iterations=5)

        ms.save_current_mesh(out_path, save_face_color=False)
        return 0
    except Exception as e:
        print("error remeshing ", file_path, "| ", e)
        return -1



if __name__=="__main__":
    original_path = paths.HOME_PATH + "data/stls/"
    new_path = paths.HOME_PATH + "data/remeshed/"
    Path(new_path).mkdir(exist_ok=True)

    directory_manager = DirectoryPathManager(original_path, base_unit_is_file=True)
    file_paths = directory_manager.file_paths

    start_from = 0
    for file_name in tqdm(file_paths[start_from:]):
        try:
            mesh = trimesh.load(file_name.as_absolute())
            mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
        except:
            print("error loading trimesh", file_name.as_relative())
            continue

        if not mesh_aux.is_valid:
            continue

        if not mesh.is_watertight:
            print("not watertight: ", file_name.as_relative())
            continue

        try:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(file_name.as_absolute())
        except:
            print("error loading meshlab", file_name.as_relative())
            continue

        try:
            ms.meshing_surface_subdivision_midpoint(iterations=3)
            ms.meshing_isotropic_explicit_remeshing(iterations=5)

            new_folder = new_path + file_name.subfolder_path
            Path(new_folder).mkdir(parents=True, exist_ok=True)
            ms.save_current_mesh(new_path + file_name.as_relative(extension=False) + ".stl", save_face_color=False)
        except Exception as e:
            print("error remeshing ", file_name.as_relative(), "| ", e)
            continue