import paths
import sklearn.model_selection
import json
import FolderManager

# Idea: create a list of train / test splits for a given folder
# Name: same as folder name. folder_splits
# These are splits based on folder names
# function: "load from split". references this file

def get_last_folder_in_path(path):
    last_folder_index = path[:-1].rfind("/") + 1
    last_folder = path[last_folder_index:]
    root_path = path[:last_folder_index]

    return last_folder, root_path

def create_splits_file(dataset_path):
    last_folder, root_path = get_last_folder_in_path(dataset_path)
    director_manager = FolderManager.DirectoryPathManager(base_path=root_path + last_folder,
                                                          base_unit_is_file=False,
                                                          recursive=False)

    all_files = director_manager.get_files_relative(extension=False)
    # all_files = paths.get_files_in_folders(base_folder="data/cad_vec/")
    train, test = sklearn.model_selection.train_test_split(all_files, test_size=0.15)
    train.sort()
    test.sort()

    save_path = root_path + last_folder[:-1] + "_splits.json"
    with open(save_path, 'w') as f:
        json.dump({
            "relative_path": last_folder,
            "train": train,
            "test": test
        }, f)

def load_mesh_folders_from_split_file(split_file, partition):
    with open(split_file, "r") as f:
        splits = json.load(f)
        return splits[partition]

if __name__=="__main__":
    # Store in upper file
    dataset_path = paths.CACHED_DATASETS_PATH + "DrivAerNet/train/"
    # relative_data_path = "train/"

    create_splits_file(dataset_path)

