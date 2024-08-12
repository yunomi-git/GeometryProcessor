

import trimesh

import paths
import trimesh_util
import numpy as np


# First grab a mesh and create random values
mesh_path = paths.HOME_PATH + "stls/squirrel.stl"
mesh = trimesh.load(mesh_path)

num_verts = len(mesh.vertices)
all_values = mesh.vertices[:, 2] # Color based on z values
all_vertex_indices = np.arange(num_verts)
# Remove some values
missing_ids = list(np.random.choice(all_vertex_indices, num_verts // 2, replace=False))

values = np.delete(all_values, missing_ids)
vertex_indices = np.delete(all_vertex_indices, missing_ids)

# Now to run the actual algorithm

def interpolate_missing_values(mesh, vertex_ids, values, max_iterations=2, iteration=0):
    # Samples may sometimes be missing values. If so, interpolate value using nearby values
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    # First create a list of values for all vertices. Missing values are nan
    input_values_padded = np.empty(len(mesh_aux.vertices))
    input_values_padded[:] = np.nan
    input_values_padded[vertex_ids] = values

    # To return
    repaired_values = np.zeros(len(mesh_aux.vertices))

    # Grab values of vertices connected to missing vertices
    vertex_connection_ids = mesh.vertex_neighbors # list
    missing_ids = np.delete(np.arange(num_verts), vertex_ids).astype(np.int32)
    missing_connection_ids = [vertex_connection_ids[i] for i in missing_ids] # list
    # Construct the missing values
    missing_values = np.empty(len(missing_ids))
    missing_values[:] = np.nan
    for i in range(len(missing_ids)):
        connections_ids = missing_connection_ids[i]
        connection_values = input_values_padded[connections_ids]
        connection_values = connection_values[~np.isnan(connection_values)]
        if len(connection_values) > 0:
            missing_values[i] = np.mean(connection_values)

    # Average and set
    repaired_values[vertex_ids] = values
    repaired_values[missing_ids] = missing_values

    if np.isnan(repaired_values).any() and iteration < max_iterations:
        # print("again")
        valid_indices = np.arange(len(mesh_aux.vertices))[~np.isnan(repaired_values)]
        repaired_values = interpolate_missing_values(mesh, vertex_ids=valid_indices,
                                                     values=repaired_values[valid_indices],
                                                     max_iterations=max_iterations, iteration=iteration+1)
    return repaired_values

repaired_values = interpolate_missing_values(mesh, vertex_indices, values)
trimesh_util.show_sampled_values(mesh, mesh.vertices[vertex_indices], values)
trimesh_util.show_sampled_values(mesh, mesh.vertices, repaired_values)
