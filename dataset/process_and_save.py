import trimesh
import trimesh_util
import paths
import numpy as np
from heuristic_prediction import printability_metrics
from tqdm import tqdm
import json
import os
from pathlib import Path

class MeshDatasetFileManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.np_mesh_base_name = "pointcloud"
        self.np_label_base_name = "target"

    def get_mesh_path(self, absolute=True):
        if absolute:
            return self.root_dir + "mesh/"
        else:
            return "mesh/"

    def get_target_path(self, absolute=True):
        if absolute:
            return self.root_dir + "target/"
        else:
            return "target/"

    def get_numpy_pointcloud_path(self, absolute=True, sampling_method="mixed"):
        if absolute:
            return self.root_dir + "numpy_pointcloud_" + sampling_method + "/"
        else:
            return "numpy_pointcloud_" + sampling_method + "/"

    def get_target_files(self, absolute=True):
        target_directory = self.get_target_path(absolute=True)
        if absolute:
            data_files = [(target_directory + file)
                          for file in os.listdir(target_directory)
                          if os.path.isfile(os.path.join(target_directory, file))]
        else:
            data_files = (file
                          for file in os.listdir(target_directory)
                          if os.path.isfile(os.path.join(target_directory, file)))
        data_files.sort()
        return data_files

    def get_mesh_files(self, absolute=True):
        target_directory = self.get_mesh_path(absolute=True)
        if absolute:
            data_files = [(target_directory + file)
                          for file in os.listdir(target_directory)
                          if os.path.isfile(os.path.join(target_directory, file))]
        else:
            data_files = (file
                          for file in os.listdir(target_directory)
                          if os.path.isfile(os.path.join(target_directory, file)))
        data_files.sort()
        return data_files

    def get_mesh_and_instances_from_target_file(self, target_path_absolute):
        with open(target_path_absolute, 'r') as f:
            target_master = json.load(f)
        mesh_path = self.root_dir + target_master["mesh_relative_path"]
        mesh = trimesh.load(mesh_path, force="mesh")

        instances = target_master["instances"]
        return mesh, instances

    def load_base_mesh_from_target(self, target_path_absolute):
        with open(target_path_absolute, 'r') as f:
            target = json.load(f)
        mesh_path = self.root_dir + target["mesh_relative_path"]
        return trimesh.load(mesh_path, force="mesh")

    def load_numpy_pointclouds(self, num_points=None, outputs_at="global", desired_label_names=None, sampling_method="mixed"):
        # outputs_at = "global", "vertices"
        # First locate the manifest
        numpy_path = self.get_numpy_pointcloud_path(absolute=True, sampling_method=sampling_method)
        manifest_path = numpy_path + "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        if outputs_at == "global":
            label_names = manifest["label_names"]
        else:
            label_names = manifest["per_point_label_names"]

        # For each file in the manifest, add to numpy array
        if num_points is None or num_points > manifest["points_per_cloud"]:
            num_points = manifest["points_per_cloud"]

        clouds = []
        targets = []
        for clouds_path in manifest["pointcloud_files"]:
            clouds_partition = np.load(numpy_path + clouds_path, allow_pickle=True)
            clouds.append(clouds_partition[:, :num_points, :])
        if outputs_at == "global":
            for targets_path in manifest["target_files"]:
                # each target is data x labels
                targets.append(np.load(numpy_path + targets_path, allow_pickle=True))
        else:
            for targets_path in manifest["per_point_target_files"]:
                # each target is data x points x labels
                target_partition = np.load(numpy_path + targets_path, allow_pickle=True)
                targets.append(target_partition[:, :num_points, :])

        clouds = np.concatenate(clouds, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Only grab the desired label names. Error if not available
        label_indices = []
        for desired_label in desired_label_names:
            label_indices.append(label_names.index(desired_label))

        if outputs_at == "global":
            targets = targets[:, label_indices]
        else:
            targets = targets[:, :, label_indices]

        # # Convert targets to dictionary
        # targets_dict = {}
        # for i in range(len(label_names)):
        #     key = label_names[i]
        #     if key not in targets_dict:
        #         targets_dict[key] = []
        #     targets_dict[key] = targets[:, i]

        return clouds, targets

# def test_load_numpy_pointclouds():
#     data_root_dir = paths.HOME_PATH + "data_th5k_aug/"
#     file_manager = MeshDatasetFileManager(data_root_dir)
#     clouds, targets = file_manager.load_numpy_pointclouds(num_points=200, desired_label_names=["stairstep_violation", "gap_violation"])
#     print(clouds.shape)
#     print(targets.shape)
#     # print(targets.keys())
#     # key1 = list(targets.keys())[0]
#     # print(targets[key1].shape)
#     return 0


def get_augmented_mesh(mesh: trimesh.Trimesh, augmentations):
    ###
    # Assumes that the mesh only contains 1 body
    # Augmentations: {
    # "euler_orientation":
    #       "x", "y", "z"
    # "scale":
    ###
    orientation = augmentations["orientation"]
    eulers = [orientation["x"], orientation["y"], orientation["z"]]
    scale = augmentations["scale"]

    transformed_mesh = trimesh_util.get_transformed_mesh(mesh, scale=scale, orientation=eulers)

    return transformed_mesh


def calculate_instance_target(mesh: trimesh.Trimesh, augmentations: dict):
    # Augmentations are just saved to the target. Not used for calculations
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    thickness_printability = printability_metrics.get_thickness_printability(mesh_aux)
    gap_printability = printability_metrics.get_gap_printability(mesh_aux)
    instance_target = {
        "vertices": mesh_aux.num_vertices,
        "edges": mesh_aux.num_edges,
        "faces": mesh_aux.num_facets,
        "bound_length": np.max(mesh_aux.bound_length),
        "is_manifold": mesh.is_watertight,
        "volume": mesh.volume,
        "overhang_violation": printability_metrics.get_overhang_printability(mesh_aux)[2],
        "stairstep_violation": printability_metrics.get_stairstep_printability(mesh_aux)[2],
        "thickness_violation": thickness_printability[2],
        "min_thickness": thickness_printability[3],
        "gap_violation": gap_printability[2],
        "min_gap": gap_printability[3],
    }
    instance_target.update(augmentations)
    return instance_target

def save_base_mesh_and_target(mesh, mesh_file_manager: MeshDatasetFileManager, mesh_name):
    # Only take the largest body in a mesh
    if mesh.body_count > 1:
        splits = list(mesh.split(only_watertight=False))
        largest_volume = 0
        largest_submesh = None
        for submesh in splits:
            temp_volume = submesh.volume
            if temp_volume > largest_volume:
                largest_volume = temp_volume
                largest_submesh = submesh
        mesh = largest_submesh

    target = {
        "base_name": mesh_name,
        "mesh_relative_path": mesh_file_manager.get_mesh_path(absolute=False) + mesh_name + ".stl",
        "instances": []
    }

    mesh.export(mesh_file_manager.get_mesh_path(absolute=True) + mesh_name + ".stl", file_type="stl")
    target_path_absolute = mesh_file_manager.get_target_path(absolute=True) + mesh_name + ".json"
    with open(target_path_absolute, 'w') as f:
        json.dump(target, f)
        
    # Add the first instance
    augmentations = {
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        },
        "scale": 1
    }
    calculate_and_append_augmentation(target=target,
                                      mesh=mesh,
                                      augmentations=augmentations)
    with open(target_path_absolute, 'w') as f:
        json.dump(target, f)
    return target, mesh



def calculate_and_append_augmentation(mesh: trimesh.Trimesh, target: dict, augmentations: dict):
    mesh = get_augmented_mesh(mesh, augmentations)
    # Normalize here if appropriate
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    # Normalization Process
    # TODO make this an option
    print("WARNING: Normalization is occurring after augmentation. calculate_and_append_augmentations()")
    centroid = np.mean(mesh_aux.vertices, axis=0)
    min_bounds = mesh_aux.bound_lower
    normalization_translation = -np.array([centroid[0], centroid[1], min_bounds[2]])
    scale = max(mesh_aux.bound_length)
    normalization_scale = 1.0 / scale
    mesh = trimesh_util.get_transformed_mesh(mesh, scale=normalization_scale, translation=normalization_translation)

    instance_target = calculate_instance_target(mesh, augmentations=augmentations)
    target["instances"].append(instance_target)


def generate_base_dataset(data_root_dir, source_mesh_filenames, save_prefix, mesh_scale=1.0, center=True,
                          normalize=True, starting_index=0):
    # Setup data path
    mesh_file_manager = MeshDatasetFileManager(root_dir=data_root_dir)
    Path(mesh_file_manager.get_mesh_path(absolute=True)).mkdir(parents=True, exist_ok=True)
    Path(mesh_file_manager.get_target_path(absolute=True)).mkdir(parents=True, exist_ok=True)

    i = starting_index
    for mesh_path in tqdm(source_mesh_filenames):
        base_name = save_prefix + "_" + "mesh" + str(i)

        mesh = trimesh.load(mesh_path, force='mesh')

        if not trimesh_util.mesh_is_valid(mesh):
            print("mesh skipped:", i)
            continue

        if not mesh.is_watertight:
            print("mesh not watertight. skipped:", i)
            continue

        mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

        normalization_scale = 1.0
        normalization_translation = np.array([0, 0, 0])
        if center:
            centroid = np.mean(mesh_aux.vertices, axis=0)
            min_bounds = mesh_aux.bound_lower
            normalization_translation = -np.array([centroid[0], centroid[1], min_bounds[2]])
        if normalize:
            scale = max(mesh_aux.bound_length)
            normalization_scale = 1.0 / scale

        mesh = trimesh_util.get_transformed_mesh(mesh, scale=normalization_scale, translation=normalization_translation)

        mesh.apply_scale(mesh_scale)
        try:
            # mesh_info, mesh_to_save = calculate_mesh_info(mesh, save_relative_path=save_mesh_relative_path, orig_mesh_path=mesh_path)
            save_base_mesh_and_target(mesh, mesh_file_manager=mesh_file_manager, mesh_name=base_name)
            i += 1
        except:
            continue

def generate_augmentations_for_dataset(mesh_file_manager: MeshDatasetFileManager, augmentation_list):
    # Setup data path
    target_path_absolute_list = mesh_file_manager.get_target_files(absolute=True)

    for target_path in tqdm(target_path_absolute_list):
        with open(target_path, "r")as f:
            target = json.load(f)

        mesh = mesh_file_manager.load_base_mesh_from_target(target_path_absolute=target_path)

        for augmentation in augmentation_list:
            calculate_and_append_augmentation(mesh=mesh, target=target, augmentations=augmentation)

        with open(target_path, 'w') as f:
            json.dump(target, f)

def quick_edit(data_root_directory):
    mesh_file_manager = MeshDatasetFileManager(data_root_directory)
    target_files = mesh_file_manager.get_target_files(absolute=True)
    for file in target_files:
        with open(file, 'r') as f:
            target = json.load(f)
        relative_path = target["mesh_relative_path"]
        period = relative_path.find(".")
        target["mesh_relative_path"] = relative_path[:period] + ".stl"
        with open(file, 'w') as f:
            json.dump(target, f)

def save_generated_dataset_as_numpy(file_manager, max_mesh_per_file, num_points_to_sample: int, sampling_method="mixed"):
    num_points_to_sample = int(num_points_to_sample)
    max_mesh_per_file = int(max_mesh_per_file)
    Path(mesh_file_manager.get_numpy_pointcloud_path(absolute=True, sampling_method=sampling_method)).mkdir(parents=True, exist_ok=True)
    all_point_clouds = []
    all_label = []
    all_pp_label = []

    # Open the base directory and get the contents
    data_files = file_manager.get_target_files(absolute=True)

    label_names = [
        "overhang_violation",
        "stairstep_violation",
        "thickness_violation",
        "gap_violation",
        "min_thickness",
        "min_gap",
        "volume",
        "bound_length"
    ]

    extra_label_names = [
        "surface_area",
    ]

    per_point_label_names = [
        "nx",
        "ny",
        "nz",
        "curvature",
        "thickness"
    ]

    # Now parse through all the files
    for data_file in tqdm(data_files):
        # Load the file
        mesh, instances = file_manager.get_mesh_and_instances_from_target_file(data_file)

        for instance_data in instances:
            mesh = get_augmented_mesh(mesh, instance_data)

            try:
                mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
                vertices, normals, face_ids = mesh_aux.sample_and_get_normals(count=int(num_points_to_sample * 1.2), # Scaling is for thickness calculation failures
                                                                          use_weight="mixed", return_face_ids=True)
                _, thicknesses, num_hits = mesh_aux.calculate_thickness_at_points(points=vertices, normals=normals)
                _, curvature = mesh_aux.calculate_curvature_at_points(origins=vertices, face_ids=face_ids,
                                                                      curvature_method="defect")

            except:
                print("sampling error")
                continue
            # np.random.shuffle(vertices)

            ## Per point labels
            # calculate thicknesses / point
            if not (num_hits >= num_points_to_sample):
                print("num hits < num samples", num_hits, num_points_to_sample)
                continue

            vertices = vertices[:num_points_to_sample]
            normals = normals[:num_points_to_sample]
            thicknesses = thicknesses[:num_points_to_sample]
            curvature = curvature[:num_points_to_sample]

            per_point_label = np.concatenate((normals, curvature[:, np.newaxis], thicknesses[:, np.newaxis]), axis=1)
            # per_point_label = normals

            assert(len(vertices) == num_points_to_sample)

            ## Global label
            label = np.array([instance_data[label_name] for label_name in label_names])
            # Append SA
            label = np.concatenate((label, np.array([mesh_aux.surface_area])))
            # Append Centroid
            label = np.concatenate((label, np.mean(vertices, axis=0)))

            ## Save
            all_point_clouds.append(vertices)
            all_label.append(label)
            all_pp_label.append(per_point_label)

    print("Total number of point clouds processed: ", len(all_point_clouds))

    # Save these
    save_path = file_manager.get_numpy_pointcloud_path(absolute=True)
    num_files_to_split = int(len(all_point_clouds) / max_mesh_per_file) + 1
    pointcloud_file_names = []
    target_file_names = []
    per_point_file_names = []
    for i in range(num_files_to_split):
        if i == num_files_to_split - 1:
            mesh_partition = all_point_clouds[i * max_mesh_per_file:]
            label_partition = all_label[i * max_mesh_per_file:]
            per_point_partition = all_pp_label[i * max_mesh_per_file:]
        else:
            mesh_partition = all_point_clouds[i * max_mesh_per_file:(i+1) * max_mesh_per_file]
            label_partition = all_label[i * max_mesh_per_file:(i+1) * max_mesh_per_file]
            per_point_partition = all_pp_label[i * max_mesh_per_file:(i+1) * max_mesh_per_file]

        mesh_partition = np.stack(mesh_partition, axis=0).astype('float32')
        label_partition = np.stack(label_partition, axis=0).astype('float32')
        per_point_partition = np.stack(per_point_partition, axis=0).astype('float32')

        pointcloud_file_name = "point_clouds_" + str(i) + ".npy"
        pointcloud_file_names.append(pointcloud_file_name)
        np.save(save_path + pointcloud_file_name, mesh_partition)

        target_file_name = "labels_" + str(i) + ".npy"
        target_file_names.append(target_file_name)
        np.save(save_path + target_file_name, label_partition)

        per_point_file_name = "per_point_" + str(i) + ".npy"
        per_point_file_names.append(per_point_file_name)
        np.save(save_path + per_point_file_name, per_point_partition)

    # also grab and save number of
    stats = {
        "pointcloud_files": pointcloud_file_names,
        "target_files": target_file_names,
        "per_point_target_files": per_point_file_names,
        "label_names": label_names + extra_label_names,
        "per_point_label_names": per_point_label_names,
        "points_per_cloud": num_points_to_sample,
        "mesh_per_file": max_mesh_per_file,
        "total_meshes": len(all_point_clouds)
    }
    with open(save_path + "manifest.json", "w") as f:
        json.dump(stats, f)

if __name__ == "__main__":
    # quick_edit(paths.HOME_PATH + "data_augmentations/")
    # 1/0
    # Generation Parameters
    outputs_save_path = paths.DATA_PATH + "data_th5k_norm/"

    mode = "convert_numpy" # generate_initial, add_augmentations, both, convert_numpy
    if mode == "generate_initial" or mode == "both":
        use_onshape = False
        source_mesh_filenames = []
        normalize_center = True
        normalize_scale = True
        if use_onshape:
            min_range = 0
            max_range = 290
            mesh_scale = 25.4
            prefix = "onshape"
            for i in range(min_range, max_range):
                source_mesh_filenames.append(paths.get_onshape_stl_path(i, get_by_order=True))
        else:
            min_range = 0
            max_range = 5000
            mesh_scale = 1.0
            prefix = "thing"
            for i in range(min_range, max_range):
                source_mesh_filenames.append(paths.get_thingiverse_stl_path(i, get_by_order=True))
    if mode == "add_augmentations" or mode == "both":
        only_add_augmentations = False
        add_basic_augmentations = True
        num_extra_augmentations = 5
    if mode == "convert_numpy":
        num_points_to_sample = 1e4
        max_mesh_per_file = 5e3
        sampling_method="even"
    overwrite = False

    # Setup data path
    mesh_file_manager = MeshDatasetFileManager(root_dir=outputs_save_path)
    Path(mesh_file_manager.get_mesh_path(absolute=True)).mkdir(parents=True, exist_ok=True)
    Path(mesh_file_manager.get_target_path(absolute=True)).mkdir(parents=True, exist_ok=True)


    if mode == "generate_initial" or mode == "both":
        generate_base_dataset(data_root_dir=outputs_save_path,
                              source_mesh_filenames=source_mesh_filenames,
                              save_prefix=prefix,
                              mesh_scale=mesh_scale,
                              normalize=normalize_scale,
                              center=normalize_center)
    if mode == "add_augmentations" or mode == "both":
        augmentation_list = []
        augmentation_list.append({
            "orientation": {
                "x": np.pi,
                "y": 0.0,
                "z": 0.0
            },
            "scale": 1
        })
        augmentation_list.append({
            "orientation": {
                "x": np.pi/2,
                "y": 0.0,
                "z": 0.0
            },
            "scale": 1
        })
        augmentation_list.append({
            "orientation": {
                "x": -np.pi/2,
                "y": 0.0,
                "z": 0.0
            },
            "scale": 1
        })
        # augmentation_list.append({
        #     "orientation": {
        #         "x": 0.0,
        #         "y": 0.0,
        #         "z": 0.0
        #     },
        #     "scale": 0.9
        # })
        # augmentation_list.append({
        #     "orientation": {
        #         "x": 0.0,
        #         "y": 0.0,
        #         "z": 0.0
        #     },
        #     "scale": 0.8
        # })

        generate_augmentations_for_dataset(mesh_file_manager=mesh_file_manager,
                                           augmentation_list=augmentation_list)

    if mode == "convert_numpy":
        save_generated_dataset_as_numpy(mesh_file_manager, max_mesh_per_file, num_points_to_sample,
                                        sampling_method=sampling_method)



