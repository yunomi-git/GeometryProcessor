import numpy as np
from abc import abstractmethod, ABC
from printability_heuristics import Thresholds
import torch

class VertexAggregator(ABC):
    @abstractmethod
    def get_name(self):
        # This goes into the json file
        pass

    @abstractmethod
    def aggregate(self, vertex_labels):
        # vertex labels is n x verts x d
        # output is n x 1
        pass

class ThicknessThresholdAggregator(VertexAggregator):
    def __init__(self, warning_thickness, failure_thickness):
        self.warning_thickness = warning_thickness
        self.failure_thickness = failure_thickness
        self.name = "Thickness_w" + str(self.warning_thickness) + "_f" + str(self.failure_thickness)
        self.loss_func = Thresholds.get_threshold_penalty(x_warn=warning_thickness, x_fail=failure_thickness, crossover=0.05, use_numpy=True)

    def get_name(self):
        return self.name

    def aggregate(self, thicknesses):
        thickness_loss = self.loss_func(thicknesses)
        loss = np.mean(thickness_loss, axis=0)
        return loss
