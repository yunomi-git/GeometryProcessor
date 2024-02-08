# if you have pyembree this should be decently fast
import matplotlib.pyplot as plt
import trimesh
import trimesh_util
import numpy as np
import paths
import random
import stopwatch
from tqdm import tqdm

def get_thickness_printability(mesh_aux: trimesh_util.MeshAuxilliaryInfo):
    points, thicknesses, num_samples = mesh_aux.calculate_thicknesses_samples(return_num_samples=True)
    poor_thickness = 0.4 # millimeters
    failure_thickness = 0.2
    violating_thicknesses = thicknesses[thicknesses < poor_thickness]
    violating_thicknesses[violating_thicknesses < failure_thickness] = failure_thickness
    violation_scores = np.square((violating_thicknesses - poor_thickness) / (poor_thickness - failure_thickness))
    normalized_violation_score = np.sum(violation_scores) / num_samples

    return points[thicknesses < poor_thickness], violation_scores, normalized_violation_score

def get_gap_printability(mesh_aux: trimesh_util.MeshAuxilliaryInfo):
    points, thicknesses, num_samples = mesh_aux.calculate_gap_samples(return_num_samples=True)
    poor_thickness = 0.4 # millimeters
    failure_thickness = 0.1
    violating_thicknesses = thicknesses[thicknesses < poor_thickness]
    violating_thicknesses[violating_thicknesses < failure_thickness] = failure_thickness
    violation_scores = np.square((violating_thicknesses - poor_thickness) / (poor_thickness - failure_thickness))
    normalized_violation_score = np.sum(violation_scores) / num_samples

    return points[thicknesses < poor_thickness], violation_scores, normalized_violation_score


def get_overhang_printability(mesh_aux: trimesh_util.MeshAuxilliaryInfo):
    minimum_angle = np.pi/4
    failure_angle = np.pi/2
    points, angles, num_samples = mesh_aux.calculate_overhangs_samples(cutoff_angle_rad=minimum_angle, return_num_samples=True)
    violating_angles = angles[angles > minimum_angle]
    violating_angles[violating_angles > failure_angle] = failure_angle
    violation_scores = np.square((violating_angles - minimum_angle) / (minimum_angle - failure_angle))
    normalized_violation_score = np.sum(violation_scores) / (num_samples / 2.0)

    return points[angles > minimum_angle], violation_scores, normalized_violation_score

def get_stairstep_printability(mesh_aux: trimesh_util.MeshAuxilliaryInfo):
    minimum_angle = np.pi/4
    failure_angle = np.pi/2
    points, angles, num_samples = mesh_aux.calculate_stairstep_samples(cutoff_angle_rad=minimum_angle, return_num_samples=True)
    violating_angles = angles[angles > minimum_angle]
    violating_angles[violating_angles > failure_angle] = failure_angle
    violation_scores = np.square((violating_angles - minimum_angle) / (minimum_angle - failure_angle))
    normalized_violation_score = np.sum(violation_scores) / (num_samples / 2.0)

    return points[angles > minimum_angle], violation_scores, normalized_violation_score

if __name__ == "__main__":
    ## Single STL
    mesh_path = paths.get_thingiverse_stl_path(5743)
    # mesh_path = paths.HOME_PATH + 'stls/crane.stl'
    mesh = trimesh.load(mesh_path)
    # mesh = trimesh_util.TRIMESH_TEST_MESH

    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    # points, values = mesh_aux.calculate_stairstep_samples(cutoff_angle_rad=np.pi/4)
    points, values, normalized_score = get_stairstep_printability(mesh_aux)
    print(1-normalized_score)

    # points, values = mesh_aux.calculate_thicknesses_samples()
    trimesh_util.show_sampled_values(mesh, points=points, values=values)

    ## Multi STL
    # for i in range(20):
    #     mesh_path = paths.get_onshape_stl_path(random.randint(1, 300))
    #     mesh = trimesh.load(mesh_path)
    #     calculate_and_show_gap(mesh)
