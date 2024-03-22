# This occurs after generating base dataset and augmentations

import paths
from pathlib import Path
import dataset.process_and_save as ps
from dataset.process_and_save import MeshDatasetFileManager

if __name__ == "__main__":
    outputs_save_path = paths.HOME_PATH + "data_th5k_aug/"

    num_points_to_sample = int(5e4)
    max_mesh_per_file = 5e3

    # Setup data path
    mesh_file_manager = MeshDatasetFileManager(root_dir=outputs_save_path)
    Path(mesh_file_manager.get_mesh_path(absolute=True)).mkdir(parents=True, exist_ok=True)
    Path(mesh_file_manager.get_target_path(absolute=True)).mkdir(parents=True, exist_ok=True)

    ps.save_generated_dataset_as_numpy(mesh_file_manager, max_mesh_per_file, num_points_to_sample)
