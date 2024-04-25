import paths
from dataset.process_and_save import MeshDatasetFileManager
from heuristic_prediction.pointcloud_dataloader import PointCloudDataset
import trimesh_util
import pyvista as pv

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
    path = paths.DATA_PATH + "data_mcb_a/"
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

def plot_all(c, r):
    pl = pv.Plotter(shape=(1, 2))
    stl_path = paths.HOME_PATH + "stls/low-res.stl"
    mesh = trimesh.load(stl_path)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    points, curvature = mesh_aux.calculate_curvature_samples(count=4096)
    pl.subplot(0, 0)
    curvature = curvature - curvature.min(axis=0)
    curvature /= curvature.max(axis=0)
    actor = pl.add_points(
        points,
        scalars=curvature,
        render_points_as_spheres=True,
        point_size=10,
        show_scalar_bar=True,
        # text="Curvature"
    )
    pl.add_text('Curvature', color='w')
    actor.mapper.lookup_table.cmap = 'jet'

    points, thickness = mesh_aux.calculate_thicknesses_samples(count=4096)
    pl.subplot(0, 1)
    thickness = thickness - thickness.min(axis=0)
    thickness /= thickness.max(axis=0)
    actor = pl.add_points(
        points,
        scalars=thickness,
        render_points_as_spheres=True,
        point_size=10,
        show_scalar_bar=True,
    )
    pl.add_text('Thickness', color='w')
    actor.mapper.lookup_table.cmap = 'jet'

    pl.link_views()
    pl.show()
