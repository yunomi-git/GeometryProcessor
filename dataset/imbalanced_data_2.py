import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from abc import ABC, abstractmethod
from tqdm import tqdm


def get_kde(data):
    min_data = np.min(data)
    max_data = np.max(data)
    width_data = max_data - min_data
    data_range = np.linspace(min_data, max_data, len(data) // 5)
    # data_range = np.linspace(min_data, max_data, 20)

    kde = KernelDensity(bandwidth=width_data / 20.0, kernel='gaussian')
    kde.fit(data[:, None])
    logprob = kde.score_samples(data_range[:, None])

    return np.exp(logprob), data_range

def get_index(value, zero_val, interval):
    index = np.ceil((value - zero_val) / interval).astype(np.int32)
    return index

def sample_equal_vertices_from_list(num_sample, data_list) -> np.ndarray:
    # data_list is data x [vertices x labels]. The number of vertices may vary across data entries, so we want to sample
    # an equal number of vertices from each entry
    # out: data x vertices x labels, all equal vertices

    num_classes = data_list[0].shape[1]
    data_sampled = []
    print("Preprocessing Data")
    for cloud in tqdm(data_list):
        cloud = cloud.copy()
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

class ImbalancedWeightingNd:
    def __init__(self, data_nd, do_classification=False):
        # data_nd as n x d for d labels per datapoint
        self.imbalanced_weighting_1d_list = []
        for i in range(data_nd.shape[1]):
            if do_classification:
                num_classes = int(np.max(data_nd[:, i])) + 1
                self.imbalanced_weighting_1d_list.append(ImbalancedWeightingClassification(data_nd[:, i], num_classes))
            else:
                self.imbalanced_weighting_1d_list.append(ImbalancedWeightingKde(data_nd[:, i]))

    def get_weights(self, values):
        weights = 0
        for imbalanced_weighting_1d in self.imbalanced_weighting_1d_list:
            weights += imbalanced_weighting_1d.get_weights(values)

        weights /= len(self.imbalanced_weighting_1d_list)
        return weights

class ImbalancedWeighting1D(ABC):
    @abstractmethod
    def get_weights(self, values):
        pass

class ImbalancedWeightingClassification(ABC):
    def __init__(self, data, num_classes):
        self.num_classes = num_classes

        self.counts = np.zeros(num_classes)
        for i in range(num_classes):
            self.counts[i] = len(data[data==i])

        self.weights = 1.0 / self.counts
        self.weights /= np.sum(self.weights)

    def get_weights(self, values):
        indices = values.astype(np.int32)

        # remove invalid indices during retrieval
        indices[indices < 0] = 0
        indices[indices >= self.num_classes] = 0

        weights = self.weights[indices]
        # max_weight = np.max(weights)

        # Set invalid indices to 0
        # weights[indices < 0] = max_weight
        # weights[indices >= self.num_classes] = max_weight
        return weights

class ImbalancedWeightingKde(ImbalancedWeighting1D):
    def __init__(self, data, max_weight=10.0):
        super(ImbalancedWeightingKde).__init__()
        self.kde, self.data_range = get_kde(data)
        self.kde /= np.max(self.kde)
        self.weights = 1.0 / (self.kde + (1.0 / max_weight))
        self.weights /= np.mean(self.weights)

        # Lookup table
        self.max_index = len(self.data_range)
        self.interval = self.data_range[1] - self.data_range[0]

    def get_weights(self, values):
        indices = get_index(values, zero_val=self.data_range[0], interval=self.interval)

        # remove invalid indices during retrieval
        indices[indices < 0] = 0
        indices[indices >= self.max_index] = 0

        weights = self.weights[indices]
        max_weight = np.max(weights)

        # Set invalid indices to 0
        weights[indices < 0] = max_weight
        weights[indices >= self.max_index] = max_weight
        return weights

def generate_random_profile(num_samples):
    num_peaks = np.random.randint(low=2, high=5)
    num_samples_per_peak = num_samples // num_peaks
    all_data = []
    for i in range(num_peaks):
        if i == num_samples - 1:
            num_samples_to_use = num_samples - (num_peaks - 1) * num_samples_per_peak
        else:
            num_samples_to_use = num_samples_per_peak
        data = np.random.normal(loc=np.random.uniform(0, 1),
                                scale=np.random.uniform(0.01, 0.05),
                                size=num_samples_to_use)
        all_data.append(data)
    all_data = np.concatenate(all_data)
    return all_data

if __name__=="__main__":
    num_samples = 10000
    data = generate_random_profile(num_samples)
    # plt.scatter(data, np.random.uniform(0, 1, num_samples), alpha=0.15)
    # plt.show()

    # kde, data_range = get_kde(data)
    # polyfit = get_poly_fit(data_range, kde)
    imbalanced_kde = ImbalancedWeightingKde(data)
    kde = imbalanced_kde.kde
    data_range = imbalanced_kde.data_range

    plt.fill_between(data_range, kde, alpha=0.5)
    plt.plot(data, np.full_like(data, -0.01), '|k', markeredgewidth=1, alpha=0.1)

    # random_values = np.random.rand(1000)
    weights = imbalanced_kde.get_weights(data_range)
    plt.scatter(data_range, weights)

    plt.fill_between(data_range, kde * imbalanced_kde.get_weights(data_range), alpha=0.5)
    plt.show()