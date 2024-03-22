import time

import numpy as np
import math
from datetime import datetime

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

def get_date_name():
    current = datetime.now()
    encode = "%d%d_%d%d" % (current.month, current.day, current.hour, current.minute)
    return encode

def get_indices_of_conditional(conditional_array):
    size = len(conditional_array)
    indices = np.arange(0, size)
    return indices[conditional_array]


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