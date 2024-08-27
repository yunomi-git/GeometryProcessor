import numpy as np
from fast_histogram import histogram1d
from scipy.special import softmax
from dataset.process_and_save import MeshDatasetFileManager
import dataset.process_and_save_temp as pas2
import paths
import bisect
import util
import matplotlib.pyplot as plt
from tqdm import tqdm



def is_not_outlier_1class(data, num_bins, threshold_ratio_to_remove):
    # Gets the indices of non-outlier data n x 1, where n is datapoints and d is labels per point.
    # returns if each element in data is an outlier or not
    data = data.flatten()
    # data as n x 1 for 1d data array
    data_min = np.min(data)
    data_max = np.max(data)
    bins = np.linspace(data_min, data_max, num=num_bins, endpoint=False)
    data_bin_list = np.array([bisect.bisect_right(a=bins, x=data[i]) - 1 for i in range(len(data))]) # which data belongs in which bin
    hist = histogram1d(data, range=[data_min, data_max*1.001], bins=num_bins)

    # which bins don't have limited data?
    threshold_num_samples_to_remove = int(threshold_ratio_to_remove * len(data))
    non_outlier_bins = util.get_indices_of_conditional(hist > threshold_num_samples_to_remove)

    return np.isin(data_bin_list, non_outlier_bins)

def non_outlier_indices(data, num_bins, threshold_ratio_to_remove):
    # Gets the indices of non-outlier data n x d, where n is datapoints and d is labels per point.
    # data as n x d for d labels per datapoint
    num_labels = data.shape[1]
    # Do 1D outlier check for each class
    is_not_outlier_per_class = np.array([is_not_outlier_1class(data[:, i], num_bins, threshold_ratio_to_remove) for i in range(num_labels)])
    is_not_outlier = np.all(is_not_outlier_per_class, axis=0)

    keep_indices = np.argwhere(is_not_outlier).flatten()
    return keep_indices


## Vertices
def vertex_to_bin_map_1class(data, num_bins):
    # Which data belongs in which bin?
    # out: data x vertices mapping vertex to bin
    # in: data in the form of data x vertices x 1 class

    # save shape as data x vertices
    shape = data.shape[:2]
    # first flatten into 1 x data*vertices
    data = data.flatten()

    min = np.min(data)
    max = np.max(data)
    bins = np.linspace(min, max, num=num_bins, endpoint=False)
    data_bin_list = np.array([bisect.bisect_right(a=bins, x=data[i]) - 1 for i in range(len(data))])

    # Then return to original shape
    return data_bin_list.reshape(shape)

def is_not_outlier_vertices_1class(data, num_bins, threshold_ratio_to_remove):
    # TODO: get this to work for inhomogeneous vertices
    # TODO: try method 2: standard deviations from median
    # each point cloud has vertices. checks for vertices that are outliers.
    # returns point clouds with no outlier vertices

    # Input: data x vertices (x 1) for 1 class
    # Output: data (x 1) boolean array

    data_min = np.min(data)
    data_max = np.max(data)

    data_bin_list = vertex_to_bin_map_1class(data, num_bins) # data x [vertices]
    hist = histogram1d(np.concatenate(data), range=[data_min, data_max*1.001], bins=num_bins)

    # which bins don't have limited data?
    non_outlier_bins = util.get_indices_of_conditional(hist > threshold_ratio_to_remove * len(data))
    not_outlier_per_cloud_vertex = np.isin(data_bin_list, non_outlier_bins) # data x vertices
    not_outlier_per_cloud = np.all(not_outlier_per_cloud_vertex, axis=1) # data (x 1)

    return not_outlier_per_cloud

def non_outlier_indices_vertices_nclass(data, num_bins, threshold_ratio_to_remove):
    # data as data x [vertices x c]. for c classes per vertex, multiple vertices per data entry
    # input is either a np array [data x vertices x c] or a list of arrays data x [vertices x c]. If latter, sample to create a np array
    # does the calculation per class. Then concatenates classes together
    num_classes = data[0].shape[1]
    num_sample = 5000
    if isinstance(data, np.ndarray):
        preprocessed_data = data
    else:
        preprocessed_data = sample_equal_vertices_from_list(num_sample=num_sample, data_list=data)

    is_not_outlier_per_class = np.array([is_not_outlier_vertices_1class(preprocessed_data[:, :, i], num_bins, threshold_ratio_to_remove) for i in range(num_classes)]) # class x data
    is_not_outlier = np.all(is_not_outlier_per_class, axis=0) # data

    keep_indices = np.argwhere(is_not_outlier).flatten()
    return keep_indices

## weights

def get_imbalanced_weight_1d(data: np.ndarray, num_bins, modifier=None):
    data = data.flatten()
    # data as n x 1 for 1d data array
    min = np.min(data)
    max = np.max(data)
    bins = np.linspace(min, max, num=num_bins, endpoint=False)
    data_bin_list = np.array([bisect.bisect_right(a=bins, x=data[i]) - 1 for i in range(len(data))])
    hist = histogram1d(data, range=[np.min(data), np.max(data)*1.001], bins=num_bins)

    # Convert the histogram to weights
    # 1/n_bin
    hist[hist > 0] = 1.0 / hist[hist > 0]
    weights_per_bin = hist
    # # (1-b^n_bin)/(1-b)
    # b = 0.99
    # weights_per_bin = (1 - np.power(b, hist)) / (1-b)


    weights_per_data = weights_per_bin[data_bin_list]
    weights_per_data /= np.mean(weights_per_data) # Normalize such that mean(w) == 1
    # weights_per_bin /= np.min(weights_per_bin[weights_per_bin > 0])
    return weights_per_data

def get_imbalanced_weight_nd(data: np.ndarray, num_bins, modifier=None):
    # data as n x d for d labels per datapoint
    num_labels = data.shape[1]
    weights_per_class = np.array([get_imbalanced_weight_1d(data[:, i], num_bins, modifier) for i in range(num_labels)]).T
    # weights_per_data = np.prod(weights_per_class, axis=0)
    return weights_per_class



def draw_data(data, num_bins):
    plt.subplot(2, 1, 1)
    plt.hist(data, range=[np.min(data), np.max(data) * 1.001], bins=num_bins)

    plt.subplot(2, 1, 2)
    if len(data.shape) < 3:
        data = data.flatten()
        plt.scatter(data, np.random.rand(len(data)), alpha=0.1)
    else: # 3
        data = data.reshape(data.shape[:2])
        indices = np.arange(len(data))
        indices = np.stack(np.repeat(indices[:, np.newaxis], repeats=data.shape[1], axis=1))
        plt.scatter(data, indices, alpha=0.1)
    plt.xlabel("Data")
    plt.show()



def draw_weights(data, weights, num_bins):
    # weight_colors = np.prod(weights, axis=1)
    weight_colors = util.normalize_minmax_01(weights)
    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    colors = cmap(weight_colors)
    colors[:, 3] = 0.05

    plt.subplot(3, 1, 1)
    plt.hist(data, range=[np.min(data), np.max(data) * 1.001], bins=num_bins)

    plt.subplot(3, 1, 2)
    if len(data.shape) < 3:
        data = data.flatten()
        plt.scatter(data, np.random.rand(len(data)), c=colors)
    else: # 3
        data = data.reshape(data.shape[:2])
        indices = np.arange(len(data))
        indices = np.stack(np.repeat(indices[:, np.newaxis], repeats=data.shape[1], axis=1))
        plt.scatter(data, indices, c=colors)
    plt.xlabel("Data")

    plt.subplot(3, 1, 3)
    plt.scatter(data, weights, c=colors)
    plt.xlabel("Data")
    plt.ylabel("Weights")
    plt.show()



## DEBUGGING
def debug_filter_outliers():
    label_names = ["SurfaceArea"]
    file_manager = pas2.DatasetManager(dataset_path=paths.CACHED_DATASETS_PATH + "th10k_norm/train/")
    _, _, data = file_manager.load_numpy_pointcloud(num_clouds=1000, num_points=1, outputs_at="global",
                                                    augmentations="none", desired_label_names=label_names)

    # label_names = ["surface_area"]
    # file_manager = MeshDatasetFileManager(root_dir=paths.DATA_PATH + "data_th5k_norm/")
    # _, data, _ = file_manager.load_numpy_pointclouds(1, outputs_at="global", desired_label_names=label_names)

    num_bins = 10

    # First look at default
    data = data.flatten()
    weights = get_imbalanced_weight_1d(data=data, num_bins=num_bins)
    draw_weights(data, weights, num_bins)

    # Then look at filtered
    # keep_indices = non_outlier_indices(data, num_bins=num_bins, threshold_ratio_to_remove=0.075)
    # data = data[keep_indices]
    keep_indices = non_outlier_indices(data, num_bins=num_bins, threshold_ratio_to_remove=0.05)
    data = data[keep_indices]
    weights = get_imbalanced_weight_1d(data=data, num_bins=num_bins)
    draw_weights(data, weights)

def sample_equal_vertices_from_list(num_sample, data_list) -> np.ndarray:
    # data_list is data x [vertices x labels]. The number of vertices may vary across data entries, so we want to sample
    # an equal number of vertices from each entry
    # out: data x vertices x labels
    num_classes = data_list[0].shape[1]
    data_sampled = []
    print("Preprocessing Data")
    for cloud in tqdm(data_list):
        sampled_cloud = np.zeros((num_sample + 2, num_classes))
        # sample num_samples from cloud. if cloud does not contain enough vertices, repeat
        if len(cloud) < num_sample:
            num_repeats = int(np.ceil(num_sample / len(cloud)))
            cloud = np.repeat(cloud, num_repeats, axis=0)
        np.random.shuffle(cloud)
        # also add min and max
        sampled_cloud[0] = np.min(cloud, axis=0)
        sampled_cloud[1] = np.max(cloud, axis=0)
        sampled_cloud[2:] = cloud[:num_sample]
        data_sampled.append(sampled_cloud)
    data_sampled = np.stack(data_sampled)
    return data_sampled

def debug_filter_outliers_vertices(load_meshes):
    label_names = ["Thickness"]
    file_manager = pas2.DatasetManager(dataset_path=paths.CACHED_DATASETS_PATH + "th10k_norm/train/")
    if load_meshes:
        _, _, _, data = file_manager.load_numpy_meshes(num_meshes=100, augmentations=None, outputs_at="vertices",
                                                       desired_label_names=label_names)
    else:
        _, _, data = file_manager.load_numpy_pointcloud(num_clouds=100, num_points=5, outputs_at="vertices",
                                                        augmentations="none", desired_label_names=label_names)
    # list of vertex values, n x [v]
    num_bins = 10

    # First look at default
    # TODO these need to work for both data = List[np] and data = np.ndarr
    print("Looking at original")
    sampled_data = sample_equal_vertices_from_list(num_sample=500, data_list=data) # TODO weights still need to be calculated across all original data
    # weights = get_imbalanced_weight_1d(data=data, num_bins=num_bins)
    flattened_data = util.flatten_list_by_one(data)
    flattened_data = np.array(flattened_data)
    flattened_data = flattened_data[:, 0]
    # flattened_data = np.random.choice(flattened_data, size=100000, replace=False)
    print("-plotting")
    draw_data(flattened_data, num_bins)
    # draw_weights(data.flatten(), weights, num_bins)
    original_length = len(data)

    # Then look at filtered
    print("Filtering")
    keep_indices = non_outlier_indices_vertices_nclass(sampled_data, num_bins=num_bins, threshold_ratio_to_remove=0.2)
    filtered_data = [data[i] for i in keep_indices]
    # data = data[keep_indices]
    # weights = get_imbalanced_weight_1d(data=data, num_bins=num_bins)
    new_length = len(filtered_data)
    flattened_data = util.flatten_list_by_one(filtered_data)
    flattened_data = np.array(flattened_data)
    flattened_data = flattened_data[:, 0]
    print("Removed", original_length - new_length, "outliers = ", (original_length - new_length)/original_length * 100, "%")
    print("-plotting")
    draw_data(flattened_data, num_bins)
    # draw_weights(data.flatten(), weights, num_bins)

def debug_draw_weights():
    # Harder test case
    file_manager = MeshDatasetFileManager(root_dir=paths.CACHED_DATASETS_PATH + "data_th5k_norm/")
    label_names = ["thickness"]
    _, data, _ = file_manager.load_numpy_pointclouds(1, outputs_at="vertices", desired_label_names=label_names)
    # data = np.array(data[:1000]).flatten()
    # data = data.flatten()
    num_bins = 10
    weights = get_imbalanced_weight_1d(data=data.flatten(), num_bins=num_bins)
    draw_weights(data, weights, num_bins)

def debug_draw_weights_vertices():
    # Harder test case
    file_manager = MeshDatasetFileManager(root_dir=paths.CACHED_DATASETS_PATH + "data_th5k_norm/")
    label_names = ["thickness"]
    _, data, _ = file_manager.load_numpy_pointclouds(1000, outputs_at="vertices", desired_label_names=label_names)
    # data = np.array(data[:1000]).flatten()
    # data is clouds x vertices x 1

    num_bins = 10
    weights = get_imbalanced_weight_1d(data=data, num_bins=num_bins)
    draw_weights(data, weights, num_bins)

def debug_print():
    # Simple test case
    data = np.array([0.1, 0.2, 2.0, 2.0, 2.0, 3, 4, 100, 100])
    num_bins = 1
    weighting = get_imbalanced_weight_1d(data=data, num_bins=num_bins)
    print(weighting)
    hist = histogram1d(data, range=[np.min(data), np.max(data) * 1.001], bins=num_bins)
    print(hist)
    hist = histogram1d(data, range=[np.min(data), np.max(data) * 1.001], bins=num_bins, weights=weighting)
    print(hist)
    # Expect all 1s

    print("---")
    # Harder test case
    file_manager = MeshDatasetFileManager(root_dir=paths.CACHED_DATASETS_PATH + "data_th5k_norm/")
    label_names = ["volume"]
    _, data, _ = file_manager.load_numpy_pointclouds(1, outputs_at="global", desired_label_names=label_names)
    # data = np.array(data[:1000]).flatten()
    data = data.flatten()
    num_bins = 1
    weighting = get_imbalanced_weight_1d(data=data, num_bins=num_bins)
    print("mean", np.mean(weighting))
    hist = histogram1d(data, range=[np.min(data), np.max(data) * 1.001], bins=num_bins)
    print(hist)
    hist = histogram1d(data, range=[np.min(data), np.max(data) * 1.001], bins=num_bins, weights=weighting)
    print(hist)
    # Expect all 1s

    print("---")
    # Harder test case
    data = np.array([[0.1, 0.2, 2.0, 2.0, 2.0, 3, 4, 100, 100],
                     [0.1, 0.2, 2.0, 2.0, 2.0, 3, 4, 100, 100]]).T
    num_bins = 10
    weighting = get_imbalanced_weight_nd(data=data, num_bins=num_bins)
    print(weighting)

if __name__=="__main__":
    debug_filter_outliers_vertices(load_meshes=True)


