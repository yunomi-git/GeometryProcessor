import time

import numpy as np
import math
from datetime import datetime
import matplotlib.pyplot as plt

def normalize_minmax_01(array: np.ndarray):
    copy = array.copy()
    copy -= np.amin(copy)
    if np.amax(copy) != 0:
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

def z_normal_to_color(direction):
    horiz_dir = np.sqrt(direction[:, 1] ** 2, direction[:, 0] ** 2)
    pitch = np.arctan2(direction[:, 2], horiz_dir) / (np.pi / 1.0) + 0.5
    # roll = np.arctan2(direction[:, 0], direction[:, 2]) / 2.0 / np.pi + 0.5
    # num_val = len(direction)
    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    colors = 255.0 * cmap(pitch)
    colors[:, 3] = int(1.0 * 255)
    return colors

def z_normal_mag_to_color(direction):
    horiz_dir = np.sqrt(direction[:, 1]**2 + direction[:, 0] ** 2)
    vert_dir = direction[:, 2]
    pitch = np.arctan2(vert_dir, horiz_dir)  # range -pi/2, pi/2
    pitch = np.abs(pitch) / (np.pi / 2) # range is 0, pi/2. change to 0, 1
    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    colors = 255.0 * cmap(pitch)
    colors[:, 3] = int(1.0 * 255)
    return colors

    # return np.stack((pitch / 2.0, pitch, pitch / 2.0, np.ones(num_val))).T

def get_date_name():
    current = datetime.now()
    encode = "%d%d_%d_%d" % (current.month, current.day, current.hour, current.minute)
    return encode

def get_indices_of_conditional(conditional_array):
    # This only works for 1d.
    size = len(conditional_array)
    indices = np.arange(0, size)
    return indices[conditional_array]

def get_indices_a_in_b(a, b):
    b_indices = []
    for a_value in a:
        b_indices.append(b.index(a_value))
    return b_indices

class DictionaryList:
    # Elements of each dictionary as a list
    def __init__(self):
        self.master_list = {}

    def add_element(self, element):
        if len(self.master_list.keys()) == 0:
            for key in element.keys():
                self.master_list[key] = []
        for key in element.keys():
            self.master_list[key].append(element[key])

if __name__=="__main__":
    values = np.arange(0, 50)
    lower_values = values < 25
    print(get_indices_of_conditional(lower_values))


class Stopwatch:
    def __init__(self):
        self.start_time = 0
        self.elapsed_time = 0

    def start(self):
        self.start_time = time.perf_counter()
        self.elapsed_time = 0

    def pause(self):
        self.elapsed_time += self.get_time()

    def resume(self):
        self.start_time = time.perf_counter()


    def print_time(self, label=""):
        print(label, time.perf_counter() - self.start_time)

    def get_time(self):
        return time.perf_counter() - self.start_time

    def get_elapsed_time(self):
        return self.elapsed_time + self.get_time()


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()