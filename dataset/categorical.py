import numpy as np
import torch

class CategoricalMap:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.num_classes = len(thresholds) + 1

    def to_category(self, x):
        if isinstance(x, list):
            cat_x = []
            for x_item in x:
                cat_x_item = np.zeros_like(x_item)
                for cat_i in range(1, self.num_classes):
                    cat_x_item[x_item > self.thresholds[cat_i - 1]] = cat_i
                cat_x.append(cat_x_item)
        else:
            # x is a numpy array
            cat_x = np.zeros_like(x)
            for cat_i in range(1, self.num_classes):
                cat_x[x > self.thresholds[cat_i - 1]] = cat_i

        return cat_x


if __name__=="__main__":
    thresholds = [0, 1, 2]
    x = np.arange(-1, 3, step=0.2)
    x = x.reshape(4, 5)

    mapping = CategoricalMap(thresholds)
    print(x)
    print(mapping.to_category(x))