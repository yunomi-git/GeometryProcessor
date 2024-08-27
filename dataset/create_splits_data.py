import os.path

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
    train, test = sklearn.model_selection.train_test_split(all_files, test_size=0.2)
    train_train, train_val = sklearn.model_selection.train_test_split(train, test_size=0.2)
    train_train.sort()
    train_val.sort()
    test.sort()

    print("train", len(train_train))
    print("val", len(train_val))
    print("test", len(test))

    save_path = get_split_file_name_for_dataset(dataset_path)
    with open(save_path, 'w') as f:
        json.dump({
            "relative_path": last_folder,
            "train": train_train,
            "validation": train_val,
            "test": test
        }, f)

def load_mesh_folders_from_split_file(dataset_path, partition, clean=False):
    if clean:
        clean_split_file(dataset_path)
    split_file = get_split_file_name_for_dataset(dataset_path)
    with open(split_file, "r") as f:
        splits = json.load(f)
        return splits[partition]

def clean_split_file(dataset_path):
    split_file = get_split_file_name_for_dataset(dataset_path)
    with open(split_file, "r") as f:
        splits = json.load(f)

    partitions = ["train", "validation", "test"]
    for partition in partitions:
        valid_folders = []
        for folder in splits[partition]:
            if os.path.exists(dataset_path + folder):
                valid_folders.append(folder)
        splits[partition] = valid_folders

    with open(split_file, 'w') as f:
        json.dump(splits, f)

def get_split_file_name_for_dataset(dataset_path):
    last_folder, root_path = get_last_folder_in_path(dataset_path)
    save_path = root_path + last_folder[:-1] + "_splits.json"
    return save_path


if __name__=="__main__":
    # Store in upper file
    dataset_path = paths.CACHED_DATASETS_PATH + "DaVinci/train/"
    # relative_data_path = "train/"

    clean_split_file(dataset_path)
    # create_splits_file(dataset_path)

