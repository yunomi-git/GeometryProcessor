import paths
from dataset.process_and_save import MeshDatasetFileManager
from heuristic_prediction.pointcloud_dataloader import PointCloudDataset
import trimesh_util

args = {
    # Dataset Param
    "num_points": 4096,
    "data_fraction": 0.3,
    "sampling_method": "even",
    "imbalanced_weighting_bins": 1, #1 means no weighting
    "remove_outlier_ratio": 0.1, # 0 means remove no outliers
    "outputs_at": "vertices"
}

label_names = ["thickness"]

if __name__=="__main__":
    path = paths.DATA_PATH + "data_primitives/"
    # file_manager = MeshDatasetFileManager(path)
    dataset = PointCloudDataset(path, args['num_points'], label_names=label_names,
                      partition='train',
                      data_fraction=args["data_fraction"], use_numpy=True,
                      sampling_method=args["sampling_method"],
                      outputs_at=args["outputs_at"],
                      imbalance_weight_num_bins=args["imbalanced_weighting_bins"],
                      remove_outlier_ratio=args["remove_outlier_ratio"])

    for i in range(len(dataset)):
        # a = dataset[i]
        cloud, labels, _ = dataset[i]
        labels = labels[:, 0]
        trimesh_util.show_sampled_values(mesh=None, points=cloud, values=labels, normalize=True)
