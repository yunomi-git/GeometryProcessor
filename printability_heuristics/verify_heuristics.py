import matplotlib.pyplot as plt
import trimesh
import trimesh_util
import numpy as np
import paths
import random
import stopwatch
from tqdm import tqdm
import shape_regression.printability_metrics as printm

def expected_sphere_overhang(min_ang):
    max_ang = np.pi/2
    return ((np.square(min_ang - max_ang) - 2) * np.sin(max_ang) + 2 * (max_ang - min_ang) * np.cos(max_ang) + 2 * np.sin(min_ang)) / np.square(max_ang - min_ang)

def expected_levels_gaps(num_levels=13, level_height=0.05, level_width=0.02, warning_length=0.4, failure_length=0.1):
    total_diameter = (num_levels + 1) * level_width

    total_area = (2 * np.pi * (total_diameter / 2) * (total_diameter / 2) + # Top and bottom
                  (num_levels + 1) * level_height * np.pi * total_diameter + # Outer cylinder
                  num_levels * level_height * np.pi * total_diameter) # Inner cylinder
    error_score = 0
    for i in range(num_levels):
        diameter = i * 2 * level_width # this is the gap
        area = np.pi * diameter * level_height
        if diameter > warning_length:
            score = 0
        elif diameter < warning_length and diameter > failure_length:
            score = ((diameter - warning_length) / (warning_length - failure_length))**2
        else: # diameter < failure_length:
            score = 1
        error_score += area * score
    return error_score / total_area

if __name__ == "__main__":
    # For sphere
    # mesh_path = paths.HOME_PATH + 'stls/sphere.stl'
    # mesh = trimesh.load(mesh_path)
    #
    # mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    #
    #
    # cutoff_angle = np.pi/4
    # _, _, overhang_metric = printm.get_overhang_printability(mesh_aux, minimum_angle=cutoff_angle, layer_height=0.0)
    # expected = expected_sphere_overhang(cutoff_angle)
    # print("expected: ", expected)
    # print("actual: ", overhang_metric)
    #
    # points, values = mesh_aux.calculate_overhangs_samples(cutoff_angle_rad=cutoff_angle, layer_height=0.0)
    # trimesh_util.show_sampled_values(mesh, points=points, values=values)

    # For levels
    mesh_path = paths.HOME_PATH + 'stls/Levels_n13_w02_h05.stl'
    mesh = trimesh.load(mesh_path)

    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)


    warning_gap = 0.4
    failure_gap = 0.1
    _, _, gap_metric = printm.get_gap_printability(mesh_aux, failure_thickness=failure_gap, warning_thickness=warning_gap)
    expected = expected_sphere_overhang(cutoff_angle)
    print("expected: ", expected)
    print("actual: ", overhang_metric)

    points, values = mesh_aux.calculate_overhangs_samples(cutoff_angle_rad=cutoff_angle, layer_height=0.0)
    trimesh_util.show_sampled_values(mesh, points=points, values=values)


