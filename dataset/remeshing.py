import trimesh
import trimesh_util
import paths
import os

#!/usr/bin/env python

"""
Remesh the input mesh to remove degeneracies and improve triangle quality.
"""


from numpy.linalg import norm

import pymesh


# if detail == "normal":
#     target_len = diag_len * 5e-3
# elif detail == "high":
#     target_len = diag_len * 2.5e-3
# elif detail == "low":
#     target_len = diag_len * 1e-2
def fix_mesh(mesh, scaling_factor, max_vertices=1e6):
    bbox_min, bbox_max = mesh.bbox
    diag_len = norm(bbox_max - bbox_min)
    target_len = diag_len * scaling_factor
    print("Target resolution: {} mm".format(target_len))

    count = 0
    mesh, __ = pymesh.remove_duplicated_vertices(mesh)
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                                               preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        print("#v: {}".format(num_vertices))
        count += 1
        if count > 5 or num_vertices > max_vertices: break

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_vertices(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    return mesh

def remesh_and_save(orig_stl_path, init_scaling_factor=4e-3, min_vertices=5e4, max_vertices=1e6):
    scaling_factor = init_scaling_factor
    temp_stl_name = "temp.stl"
    # then load into pymesh and fix
    print("remeshing")
    pymesh_mesh = pymesh.meshio.load_mesh(orig_stl_path)
    fixed_pymesh_mesh = fix_mesh(pymesh_mesh, scaling_factor)
    print("Num vertices: ", pymesh_mesh.num_vertices)

    # Check if it is fine
    # Check if reached idea number of vertices, and adjust accordingly
    # check if number of bodies increased
    for i in range(2):
        pymesh.meshio.save_mesh(temp_stl_name, fixed_pymesh_mesh)
        fixed_trimesh_mesh = trimesh.load(temp_stl_name)
        if fixed_trimesh_mesh.body_count > 1:
            print("too many bodies")
            scaling_factor /= 1.2
            # trimesh_util.show_mesh(fixed_trimesh_mesh)
        elif fixed_pymesh_mesh.num_vertices < min_vertices:
            print("Not enough vertices")
            scaling_factor /= 1.5
            # trimesh_util.show_mesh(fixed_trimesh_mesh)
        elif fixed_pymesh_mesh.num_vertices > max_vertices:
            print("Too many vertices")
            scaling_factor *= 1.5
            # trimesh_util.show_mesh(fixed_trimesh_mesh)
        else:
            break # All checks passed. now save
        pymesh_mesh = pymesh.meshio.load_mesh(orig_stl_path)
        fixed_pymesh_mesh = fix_mesh(pymesh_mesh, scaling_factor)
        print("Num vertices: ", len(fixed_trimesh_mesh.vertices))

    # Load trimesh and grab larges body
    fixed_trimesh_mesh = trimesh.load(temp_stl_name)
    if fixed_trimesh_mesh.body_count > 1:
        splits = list(fixed_trimesh_mesh.split(only_watertight=False))
        largest_volume = 0
        largest_submesh = None
        for submesh in splits:
            temp_volume = submesh.volume
            if temp_volume > largest_volume:
                largest_volume = temp_volume
                largest_submesh = submesh
        fixed_trimesh_mesh = largest_submesh

    # Now save
    # fixed_trimesh_mesh.export(new_stl_path)
    os.remove(temp_stl_name)
    return fixed_trimesh_mesh


if __name__ == "__main__":
    # Single STL
    mesh_path = paths.get_onshape_stl_path(233)
    # mesh_path = paths.get_thingiverse_stl_path(258, get_by_order=True)
    # mesh_path = paths.HOME_PATH + 'stls/crane.stl'

    # This is the original
    mesh = trimesh.load(mesh_path)
    trimesh_util.show_mesh(mesh)
    #
    # # This is the editted
    # mesh = pymesh.meshio.load_mesh(mesh_path)
    # mesh = fix_mesh(mesh, scaling_factor=4e-3)
    # mesh_save_path = paths.HOME_PATH + "temp.stl"
    # pymesh.meshio.save_mesh(mesh_save_path, mesh)
    #
    # mesh_new = trimesh.load(mesh_save_path)
    # trimesh_util.show_mesh(mesh_new)

    fixed_mesh = remesh_and_save(mesh_path)
    print("Num vertices: ", len(fixed_mesh.vertices))
    trimesh_util.show_mesh(fixed_mesh)
