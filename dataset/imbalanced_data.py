import numpy as np
from fast_histogram import histogram1d
from scipy.special import softmax
from dataset.process_and_save import MeshDatasetFileManager
import paths
import bisect
import util
import matplotlib.pyplot as plt




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

    # Input: data x vertices x 1 for 1 class
    # Output: data x 1 boolean array

    data_min = np.min(data)
    data_max = np.max(data)

    data_bin_list = vertex_to_bin_map_1class(data, num_bins) # data x [vertices]
    hist = histogram1d(data.flatten(), range=[data_min, data_max*1.001], bins=num_bins)

    # which bins don't have limited data?
    non_outlier_bins = util.get_indices_of_conditional(hist > threshold_ratio_to_remove * len(data))
    not_outlier_per_cloud_vertex = np.isin(data_bin_list, non_outlier_bins) # data x vertices
    not_outlier_per_cloud = np.all(not_outlier_per_cloud_vertex, axis=1) # data x 1

    return not_outlier_per_cloud

def non_outlier_indices_vertices_nclass(data, num_bins, threshold_ratio_to_remove):
    # data as data x vertices x c. for c classes per vertex, multiple vertices per datapoint
    # does the calculation per class. Then concatenates classes together

    num_labels = data.shape[2]
    is_not_outlier_per_class = np.array([is_not_outlier_vertices_1class(data, num_bins, threshold_ratio_to_remove) for i in range(num_labels)]) # class x data
    is_not_outlier = np.all(is_not_outlier_per_class, axis=0) # data

    keep_indices = np.argwhere(is_not_outlier).flatten()
    return keep_indices

## weights

def get_imbalanced_weight_1d(data, num_bins, modifier=None):
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

def get_imbalanced_weight_nd(data, num_bins, modifier=None):
    # data as n x d for d labels per datapoint
    num_labels = data.shape[1]
    weights_per_class = np.array([get_imbalanced_weight_1d(data[:, i], num_bins, modifier) for i in range(num_labels)]).T
    # weights_per_data = np.prod(weights_per_class, axis=0)
    return weights_per_class

def draw_weights(data, weights):
    # weight_colors = np.prod(weights, axis=1)
    weight_colors = util.normalize_minmax_01(weights)
    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    colors = cmap(weight_colors)
    colors[:, 3] = 0.15

    plt.subplot(2, 1, 1)
    if len(data.shape) < 3:
        data = data.flatten()
        plt.scatter(data, np.random.rand(len(data)), c=colors)
    else: # 3
        data = data.reshape(data.shape[:2])
        indices = np.arange(len(data))
        indices = np.stack(np.repeat(indices[:, np.newaxis], repeats=data.shape[1], axis=1))
        plt.scatter(data, indices, c=colors)
    plt.xlabel("Data")

    plt.subplot(2, 1, 2)
    plt.scatter(data, weights, c=colors)
    plt.xlabel("Data")
    plt.ylabel("Weights")
    plt.show()



## DEBUGGING
def debug_filter_outliers():
    file_manager = MeshDatasetFileManager(root_dir=paths.DATA_PATH + "data_th5k_norm/")
    label_names = ["surface_area"]
    _, data, _ = file_manager.load_numpy_pointclouds(1, outputs_at="global", desired_label_names=label_names)
    num_bins = 10

    # First look at default
    # data = data.flatten()
    weights = get_imbalanced_weight_1d(data=data, num_bins=num_bins)
    draw_weights(data, weights)

    # Then look at filtered
    # keep_indices = non_outlier_indices(data, num_bins=num_bins, threshold_ratio_to_remove=0.075)
    # data = data[keep_indices]
    keep_indices = non_outlier_indices(data, num_bins=num_bins, threshold_ratio_to_remove=0.05)
    data = data[keep_indices]
    weights = get_imbalanced_weight_1d(data=data, num_bins=num_bins)
    draw_weights(data, weights)

def debug_filter_outliers_vertices():
    file_manager = MeshDatasetFileManager(root_dir=paths.DATA_PATH + "data_th5k_norm/")
    label_names = ["thickness"]
    _, data, _ = file_manager.load_numpy_pointclouds(5, outputs_at="vertices", desired_label_names=label_names)
    num_bins = 10

    # First look at default
    weights = get_imbalanced_weight_1d(data=data, num_bins=num_bins)
    draw_weights(data.flatten(), weights)
    original_length = len(data)

    # Then look at filtered
    keep_indices = non_outlier_indices_vertices_nclass(data, num_bins=num_bins, threshold_ratio_to_remove=0.01)
    data = data[keep_indices]
    weights = get_imbalanced_weight_1d(data=data, num_bins=num_bins)
    new_length = len(data)
    print("Removed", original_length - new_length, "outliers = ", (original_length - new_length)/original_length * 100, "%")
    draw_weights(data, weights)

def debug_draw_weights():
    # Harder test case
    file_manager = MeshDatasetFileManager(root_dir=paths.DATA_PATH + "data_th5k_norm/")
    label_names = ["thickness"]
    _, data, _ = file_manager.load_numpy_pointclouds(1, outputs_at="vertices", desired_label_names=label_names)
    # data = np.array(data[:1000]).flatten()
    # data = data.flatten()
    num_bins = 10
    weights = get_imbalanced_weight_1d(data=data.flatten(), num_bins=num_bins)
    draw_weights(data, weights)

def debug_draw_weights_vertices():
    # Harder test case
    file_manager = MeshDatasetFileManager(root_dir=paths.DATA_PATH + "data_th5k_norm/")
    label_names = ["thickness"]
    _, data, _ = file_manager.load_numpy_pointclouds(1000, outputs_at="vertices", desired_label_names=label_names)
    # data = np.array(data[:1000]).flatten()
    # data is clouds x vertices x 1

    num_bins = 10
    weights = get_imbalanced_weight_1d(data=data, num_bins=num_bins)
    draw_weights(data, weights)

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
    file_manager = MeshDatasetFileManager(root_dir=paths.DATA_PATH + "data_th5k_norm/")
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
    debug_filter_outliers_vertices()


