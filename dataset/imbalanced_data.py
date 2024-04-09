import numpy as np
from fast_histogram import histogram1d
from scipy.special import softmax
from dataset.process_and_save import MeshDatasetFileManager
import paths
import bisect
import util
import matplotlib.pyplot as plt

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
    cmap = plt.get_cmap(cmapname)  # TODO NOTE: these are out of order!
    colors = cmap(weight_colors)
    colors[:, 3] = 0.15

    plt.subplot(2, 1, 1)
    plt.scatter(data, np.random.rand(len(data)), c=colors)
    plt.xlabel("Data")

    plt.subplot(2, 1, 2)
    plt.scatter(data, weights, c=colors)
    plt.xlabel("Data")
    plt.ylabel("Weights")
    plt.show()

def debug_draw_weights():
    # Harder test case
    file_manager = MeshDatasetFileManager(root_dir=paths.DATA_PATH + "data_th5k_norm/")
    label_names = ["thickness"]
    _, data = file_manager.load_numpy_pointclouds(1, outputs_at="vertices", desired_label_names=label_names)
    # data = np.array(data[:1000]).flatten()
    data = data.flatten()
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
    _, data = file_manager.load_numpy_pointclouds(1, outputs_at="global", desired_label_names=label_names)
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
    debug_draw_weights()


