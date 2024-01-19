import trimesh
import numpy as np
import time
from stopwatch import Stopwatch
import trimesh_util
import paths
 # https://towardsdatascience.com/how-to-voxelize-meshes-and-point-clouds-in-python-ca94d403f81d
stopwatch = Stopwatch()
directions = ['x', 'y', 'z']
inv_directions = {
    'x': 0,
    'y': 1,
    'z': 2
}
inv_sign = {
    '-1': 0,
    '1': 1
}

wall_indices_map = {
    "+x": (1, 0),
    "-x": (0, 0),
    "+y": (1, 1),
    "-y": (0, 1),
    "+z": (1, 2),
    "-z": (0, 2),
}



def get_closest_intersecting_bound(mesh, facet_i):
    # centroid = mesh.facet_origin[facet_i, :]
    # normal = mesh.facet_normal[facet_i, :]
    # min_bounds = mesh.bounds[0, :]
    # max_bounds = mesh.bounds[1, :]
    # sub_bounds = np.empty(3)
    #
    # if normal[0] < 0:
    #     sub_bounds[0] = min_bounds[0] - centroid[0]
    # else:
    #     sub_bounds[0] = centroid[0] - max_bounds[0]
    #
    # normalized_normals = np.empty(3)
    # for j in range(3):
    #     normalized_normals = normal[j] / sub_bounds[j]
    return "+z"

def interpolate_3d(start, end, ratio):
    return start + ratio * (end - start)

def search_for_surface_point(voxel, start_point, end_point):
    filled_bound = 0.0
    unfilled_bound = 1.0
    for i in range(8):
        test_ratio = (filled_bound + unfilled_bound) / 2.0

        stopwatch.start()
        test_point = interpolate_3d(start_point, end_point, ratio=test_ratio)
        # point_is_filled = voxel.is_filled(test_point)
        point_is_filled = trimesh_util.check_voxel_is_filled(voxel, point=test_point)
        print("voxel fill check", point_is_filled)
        stopwatch.get_time()

        if point_is_filled:
            filled_bound = test_point
        else:
            unfilled_bound = test_point
    return interpolate_3d(start_point, end_point, unfilled_bound)

def find_boundary_intersection(origin, direction, wall, bounds):
    # Find z intersection
    wall_indices = wall_indices_map[wall]
    z_intersect = bounds[wall_indices[0], wall_indices[1]]
    z_travel = z_intersect - origin[2]
    x_intersect = origin[0] + direction[0] / direction[2] * z_travel
    y_intersect = origin[1] + direction[1] / direction[2] * z_travel
    intersect = np.array([x_intersect, y_intersect, z_intersect])
    return intersect

def get_thickness_from_voxels(mesh, voxel):

    num_faces = len(mesh.faces)
    face_thicknesses = np.empty(num_faces)

    for i in range(num_faces):
        print(i)
        origin = mesh.triangles_center[i, :]
        closest_bound = get_closest_intersecting_bound(mesh, i)
        # find intersection with closest bound
        bound_intersection_point = find_boundary_intersection(origin, -mesh.face_normals[i,:], closest_bound, mesh.bounds)

        end_point = bound_intersection_point
        start_point = origin

        stopwatch.start()
        surface_point = search_for_surface_point(voxel, start_point, end_point)
        print("search intersect")
        stopwatch.get_time()


        face_thicknesses[i] = np.linalg.norm(surface_point - start_point)

    return face_thicknesses

if __name__=="__main__":
    # mesh_path = 'stls/low-res.stl'
    # mesh = trimesh.load(mesh_path)
    mesh = trimesh_util.TRIMESH_TEST_MESH
    # mesh_path = paths.STL_PATH + "solid_1.stl"
    voxels = trimesh_util.voxelize(mesh)
    facet_thicknesses = get_thickness_from_voxels(mesh, voxels)

    # Now with voxels, do ray tracing
    mesh.visual.face_colors = map(facet_thicknesses)


    s = trimesh.Scene()
    s.add_geometry(mesh)
    # Add the voxelized mesh to the scene. If want to also show the intial mesh uncomment the second line and change the alpha channel of in the loop to something <100
    # s.add_geometry(voxels)
    s.show()

