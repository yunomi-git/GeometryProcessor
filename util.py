import numpy as np
import math

def normalize_minmax_01(array: np.ndarray):
    copy = array.copy()
    copy -= np.amin(copy)
    copy /= np.amax(copy)
    return copy

def normalize_max_1(array: np.ndarray):
    copy = array.copy()
    minim = np.amin(copy)
    if minim < 0:
        copy -= minim
    copy /= np.amax(copy)
    return copy

def direction_to_color(direction):
    # Yaw and pitch
    # Yaw: project onto z
    yaw = np.arctan2(direction[:, 1], direction[:, 0]) / 2.0 / np.pi + 0.5
    pitch = np.arctan2(direction[:, 2], direction[:, 1]) / 2.0 / np.pi + 0.5
    roll = np.arctan2(direction[:, 0], direction[:, 2]) / 2.0 / np.pi + 0.5
    num_val = len(direction)
    return np.stack((yaw, pitch, roll, np.ones(num_val))).T



