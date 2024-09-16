import numpy as np
import torch


def get_threshold_penalty(x_warn, x_fail, crossover=0.05, use_numpy=False):
    # Function is 1/ 1 + exp(-a (x + b)). We need to solve for a and b
    cross_warn = np.log(1.0 / crossover - 1)
    cross_fail = np.log(1.0 / (1 - crossover) - 1)
    a = (cross_warn - cross_fail) / (x_fail - x_warn)
    b = (cross_warn * x_fail - cross_fail * x_warn) / (cross_fail - cross_warn)
    if use_numpy:
        penalty_func = lambda x: 1.0 / (1.0 + np.exp(-a * (x + b)))
    else:
        penalty_func = lambda x: torch.sigmoid(a * (x + b))
        # penalty_func = lambda x: 1.0 / (1.0 + torch.exp(-a * (x + b)))
    return penalty_func
