import paths
from dataset.process_and_save import MeshDatasetFileManager
from heuristic_prediction.pointcloud_dataloader import PointCloudDataset
import trimesh_util
import pyvista as pv
import numpy as np

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

def plot_all(c, r, dataset):
    num_data = len(dataset)
    num_visualized = c * r
    num_iterations = int(np.ceil(num_data / num_visualized))

    for i in range(num_iterations):
        pl = pv.Plotter(shape=(r, c))
        for ri in range(r):
            for ci in range(c):
                cloud, _, _ = dataset[i * r * c + ri * c + ci]
                pl.subplot(ri, ci)
                actor = pl.add_points(
                    points=cloud,
                    # scalars=curvature,
                    render_points_as_spheres=True,
                    point_size=10,
                    show_scalar_bar=True,
                    # text="Curvature"
                )
                # actor.mapper.lookup_table.cmap = 'jet'
                pl.show_bounds(grid=True, all_edges=True,  font_size=10)


        pl.link_views()
        pl.show()

if __name__=="__main__":
    path = paths.DATA_PATH + "mcb_scale_a/"
    # file_manager = MeshDatasetFileManager(path)
    dataset = PointCloudDataset(path, args['num_points'], label_names=label_names,
                      partition='train',
                      data_fraction=args["data_fraction"], use_numpy=True,
                      sampling_method=args["sampling_method"],
                      outputs_at=args["outputs_at"],
                      imbalance_weight_num_bins=args["imbalanced_weighting_bins"],
                      remove_outlier_ratio=args["remove_outlier_ratio"])

    all_clouds = dataset.point_clouds
    # data x point x position


    print("avg_position: ", np.mean(np.mean(dataset.point_clouds, axis=1), axis=0))
    print("min_position: ", np.min(np.min(dataset.point_clouds, axis=1), axis=0))
    print("max_position: ", np.max(np.max(dataset.point_clouds, axis=1), axis=0))
    lengths = np.max(dataset.point_clouds, axis=1) - np.min(dataset.point_clouds, axis=1)

    print("avg_length: ", np.mean(lengths, axis=0))
    print("min_length: ", np.min(lengths, axis=0))
    print("max_length: ", np.max(lengths, axis=0))

    plot_all(5, 6, dataset)
    # for i in range(len(dataset)):
    #     # a = dataset[i]
    #     cloud, labels, _ = dataset[i]
    #     labels = labels[:, 0]
    #     trimesh_util.show_sampled_values(mesh=None, points=cloud, values=labels, normalize=True)


