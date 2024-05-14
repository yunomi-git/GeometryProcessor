import torch

from Generation.Thresholds import get_threshold_penalty
import matplotlib.pyplot as plt
import numpy as np

def get_function(function, range):
    return function(torch.from_numpy(range)).numpy()



if __name__=="__main__":
    # thresh_1 = get_threshold_penalty(x_warn=0.2, x_fail=0.95, crossover=0.01)
    # thresh_2 = get_threshold_penalty(x_warn=1.0, x_fail=0.90, crossover=0.01)
    # thresh_1 = get_threshold_penalty(x_warn=torch.pi / 4, x_fail=torch.pi / 2 * 0.95, crossover=0.01)
    # thresh_2 = get_threshold_penalty(x_warn=torch.pi/2*0.95, x_fail=torch.pi / 2 * 0.90, crossover=0.01)
    # range = np.linspace(-np.pi/2, np.pi/2, num=500)

    # thresh_1 = get_threshold_penalty(x_warn=-torch.pi / 2, x_fail=-torch.pi / 4, crossover=0.01)
    # range = np.linspace(-np.pi/2, np.pi/2, num=500)


    min_height = 0
    layer_height = 0.2
    thresh_1 = get_threshold_penalty(x_warn=min_height, x_fail=min_height + layer_height, crossover=0.01)
    range = np.linspace(0, 1, num=500)

    plt.plot(range, get_function(thresh_1, range))
    # plt.plot(range, get_function(thresh_2, range))
    # plt.plot(range, get_function(thresh_1, range) * get_function(thresh_2, range))

    # print(thresh_1(torch.from_numpy(
    #                         np.array([0.2])))
    #       .numpy())
    # print(thresh_1(torch.from_numpy(
    #                         np.array([0.95])))
    #       .numpy())
    plt.show()
