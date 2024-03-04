# if you have pyembree this should be decently fast
import matplotlib.pyplot as plt
import trimesh
import trimesh_util
import numpy as np
import paths
import random
import stopwatch
from tqdm import tqdm


def get_thickness_printability(mesh_aux: trimesh_util.MeshAuxilliaryInfo,
                               warning_thickness=0.4,
                               failure_thickness=0.2):
    points, thicknesses, num_samples = mesh_aux.calculate_thicknesses_samples(return_num_samples=True)
    if num_samples == 0:
        return [], np.empty(1), 0.0, 0.0
    violating_thicknesses = thicknesses[thicknesses < warning_thickness]
    violating_thicknesses[violating_thicknesses < failure_thickness] = failure_thickness
    violation_scores = np.square((violating_thicknesses - warning_thickness) / (warning_thickness - failure_thickness))
    normalized_violation_score = np.sum(violation_scores) / num_samples
    if len(violating_thicknesses) == 0:
        min_thickness = trimesh_util.NO_GAP_VALUE
    else:
        min_thickness = np.min(violating_thicknesses)

    return points[thicknesses < warning_thickness], violation_scores, normalized_violation_score, min_thickness


def get_gap_printability(mesh_aux: trimesh_util.MeshAuxilliaryInfo,
                         warning_thickness=0.4,
                         failure_thickness=0.1):
    points, thicknesses, num_samples = mesh_aux.calculate_gap_samples(return_num_samples=True)
    violating_thicknesses = thicknesses[thicknesses < warning_thickness]
    if len(violating_thicknesses) == 0:
        return [], np.empty(1), 0.0, 0.0

    violating_thicknesses[violating_thicknesses < failure_thickness] = failure_thickness
    violation_scores = np.square((violating_thicknesses - warning_thickness) / (warning_thickness - failure_thickness))
    normalized_violation_score = np.sum(violation_scores) / num_samples
    if len(violating_thicknesses) == 0:
        min_gap = -1.0
    else:
        min_gap = np.min(violating_thicknesses)

    return points[thicknesses < warning_thickness], violation_scores, normalized_violation_score, min_gap


def get_overhang_printability(mesh_aux: trimesh_util.MeshAuxilliaryInfo,
                              minimum_angle=np.pi/4):
    maximum_angle = np.pi/2
    points, angles, num_samples = mesh_aux.calculate_overhangs_samples(cutoff_angle_rad=minimum_angle, return_num_samples=True)
    violating_angles = angles[angles > minimum_angle]
    violating_angles[violating_angles > maximum_angle] = maximum_angle
    violation_scores = np.square((violating_angles - minimum_angle) / (minimum_angle - maximum_angle))
    normalized_violation_score = np.sum(violation_scores) / (num_samples / 2.0)

    return points[angles > minimum_angle], violation_scores, normalized_violation_score


def get_stairstep_printability(mesh_aux: trimesh_util.MeshAuxilliaryInfo,
                               minimum_angle=np.pi/4,
                               maximum_angle=np.pi/2 * 0.90):
    points, angles, num_samples = mesh_aux.calculate_stairstep_samples(min_angle_rad=minimum_angle,
                                                                       max_angle_rad=maximum_angle,
                                                                       return_num_samples=True)
    violating_indices = np.logical_and(angles > minimum_angle, angles < maximum_angle)
    violating_angles = angles[violating_indices]
    violation_scores = np.square((violating_angles - minimum_angle) / (minimum_angle - maximum_angle))
    normalized_violation_score = np.sum(violation_scores) / (num_samples / 2.0)

    return points[angles > minimum_angle], violation_scores, normalized_violation_score

# Issues: onshape 202, thing 285, onshape 37, onehape 89
if __name__ == "__main__":
    ## Single STL
    # mesh_path = paths.get_thingiverse_stl_path(5743)
    # mesh_path = paths.HOME_PATH + 'stls/crane.stl'
    mesh_path = paths.TRAINING_DATA_PATH + "mesh/onshape_mesh202.stl"
    mesh = trimesh.load(mesh_path)
    # mesh = trimesh_util.TRIMESH_TEST_MESH

    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    # points, values = mesh_aux.calculate_stairstep_samples(cutoff_angle_rad=np.pi/4)
    points, values, normalized_score = get_thickness_printability(mesh_aux)
    print(1-normalized_score)

    # points, values = mesh_aux.calculate_thicknesses_samples()
    trimesh_util.show_sampled_values(mesh, points=points, values=values)

    ## Multi STL
    # for i in range(20):
    #     mesh_path = paths.get_onshape_stl_path(random.randint(1, 300))
    #     mesh = trimesh.load(mesh_path)
    #     calculate_and_show_gap(mesh)
