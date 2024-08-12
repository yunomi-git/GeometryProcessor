# Contents of a folder


import os
from typing import List

import paths
from tqdm import tqdm

class FilePath:
    def __init__(self, base_path, subfolder_path, file_name, is_folder):
        if is_folder:
            self.extension = "/"
            self.file_name = file_name
        else:
            self.extension = file_name[file_name.find("."):]
            self.file_name = file_name[:file_name.find(".")]
        self.subfolder_path = subfolder_path
        self.base_path = base_path
        if len(subfolder_path) == 0:
            self.subfolder_split = []
        else:
            self.subfolder_split = self.subfolder_path[:-1].split("/")

        # self.subfolder_split = [folder if len(folder) > 0 for folder in self.subfolder_split]

    def as_subfolder_string(self):
        name = ""
        for subfolder in self.subfolder_split:
            name += subfolder + "_"
        name += self.file_name
        return name

    def as_absolute(self):
        return self.base_path + self.subfolder_path + self.file_name + self.extension

    def as_relative(self, extension=True):
        if extension:
            return self.subfolder_path + self.file_name + self.extension
        else:
            return self.subfolder_path + self.file_name

class DirectoryPathManager:
    def __init__(self, base_path, base_unit_is_file, max_files_per_subfolder=-1, recursive=True):
        self.base_path = base_path
        self.recursive = recursive
        self.base_unit_is_file = base_unit_is_file

        max_depth = -1
        if not recursive:
            max_depth = 0

        if base_unit_is_file:
            self.file_paths = get_all_files_in_directory(base_path,
                                                         max_files_per_subfolder=max_files_per_subfolder,
                                                         max_depth=max_depth)
        else:
            self.file_paths = get_all_final_folders_in_directory(base_path, max_depth=max_depth)


    def get_files_absolute(self) -> List[str]:
        return [file.as_absolute() for file in self.file_paths]

    def get_files_relative(self, extension=True) -> List[str]:
        return [file.as_relative(extension) for file in self.file_paths]

    def get_file_names(self, extension=True):
        if extension:
            return [file.file_name + file.extension for file in self.file_paths]
        else:
            return [file.file_name for file in self.file_paths]

def folder_contains_no_folders(folder_path_absolute):
    contents = os.listdir(folder_path_absolute)
    for content in contents:
        if os.path.isdir(folder_path_absolute + content):
            return False
    return True

def get_all_final_folders_in_directory(base_path, subfolder_path="", max_depth=-1, depth=0):
    base_contents = os.listdir(base_path + subfolder_path)
    base_contents.sort()

    folder_paths = []
    for content in base_contents:
        if not os.path.isdir(base_path + subfolder_path + content):
            continue
        if folder_contains_no_folders(base_path + subfolder_path + content + "/") or (max_depth != -1 and depth == max_depth):
            folder_paths.append(FilePath(base_path, subfolder_path, content, is_folder=True))
        else:
            if max_depth != -1 and depth < max_depth:
                folder_paths.extend(get_all_final_folders_in_directory(base_path,
                                                                       subfolder_path=subfolder_path + content + "/",
                                                                       max_depth=max_depth,
                                                                       depth=depth+1))
    return folder_paths

def get_all_files_in_directory(base_path, subfolder_path="", max_files_per_subfolder=-1, max_depth=-1, depth=0) -> List[FilePath]:
    base_contents = os.listdir(base_path + subfolder_path)
    base_contents.sort()
    if max_files_per_subfolder > 0:
        max_files = len(base_contents)
        if max_files_per_subfolder < max_files:
            max_files = max_files_per_subfolder
        base_contents = base_contents[:max_files]
    files_paths = []
    for content in tqdm(base_contents):
        if os.path.isfile(base_path + subfolder_path + content):
            files_paths.append(FilePath(base_path, subfolder_path, content, is_folder=False))
        elif max_depth != -1 and depth < max_depth:
            files_paths.extend(get_all_files_in_directory(base_path,
                                                          subfolder_path=subfolder_path + content + "/",
                                                          max_files_per_subfolder=max_files_per_subfolder,
                                                          max_depth=max_depth,
                                                          depth=depth+1))
    return files_paths



if __name__=="__main__":
    base_path = paths.HOME_PATH + "../Datasets/MCB_A/"
    directory_manager = DirectoryPathManager(base_path=base_path, max_files_per_subfolder=100,
                                             base_unit_is_file=True)
    absolutes = directory_manager.get_files_absolute()
    relative = directory_manager.get_files_relative(extension=False)

    folder_directory_manager = DirectoryPathManager(base_path=base_path, max_files_per_subfolder=100,
                                                    base_unit_is_file=False)
    folder_absolutes = folder_directory_manager.get_files_absolute()
    folder_relative = folder_directory_manager.get_files_relative(extension=False)

    print("a")