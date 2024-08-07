import trimesh
import bpy
import bmesh

import paths
import trimesh_util
import numpy as np
from blender_util import BlenderTrimeshManager
mesh = trimesh.load(paths.HOME_PATH + 'objs/bunny_pancake_image_sai_custom_1024/output_mesh.obj')
mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
# trimesh_util.show_mesh(mesh)
visual = mesh.visual
num_vertices = len(mesh.vertices)
image = visual.material.image

# trimesh_util.show_mesh(mesh)

vertex_colors = np.zeros((num_vertices, 4))
for i in range(num_vertices):
    uv = visual.vertex_attributes.data['uv'][i]
    x = int(image.size[0] * uv[0])
    y = int(image.size[1] * uv[1])
    # print(position)
    color = image.getpixel((x, y))
    vertex_colors[i, :3] = color
    vertex_colors[i, 3] = 255

vertex_grayscale = np.dot(vertex_colors[..., :3], [0.2989, 0.5870, 0.1140]) / 255.0
vertex_grayscale = 1 - vertex_grayscale
vertex_grayscale -= min(vertex_grayscale)
vertex_grayscale /= max(vertex_grayscale)
vertex_normals = mesh.vertex_normals
scale = min(mesh_aux.bound_length) * 0.04

vertex_diff = - np.stack((vertex_grayscale, vertex_grayscale, vertex_grayscale), axis=-1) * scale * vertex_normals
# mesh.vertices += vertex_diff
# trimesh_util.show_mesh(mesh)

# mesh.export("temp.obj")
#
# s = trimesh.Scene()
vertices = mesh.vertices
# point_cloud = trimesh.points.PointCloud(vertices=vertices,
#                                         colors=vertex_colors)
# s.add_geometry(point_cloud)
# s.show()


# mesh_manager = BlenderTrimeshManager(mesh)

# Grab vertices in bmesh mode
# blender_verts = mesh_manager.get_blender_verts_from_indices(list(range(num_vertices)))
# for i in range(num_vertices):
#     vert = blender_verts[i]
#     vert_diff_mag = vertex_diff[i]
#     bmesh.ops.translate(mesh_manager.blender_mesh, verts=[vert],
#                         vec=vert_diff_mag)

# bmesh.ops.smooth_vert(mesh_manager.blender_mesh, verts=mesh_manager.blender_mesh.verts, factor=0.25,
#                       use_axis_x=True, use_axis_y=True, use_axis_z=True)

# mesh_manager.update_trimesh_from_blender()
# trimesh_util.show_mesh(mesh_manager.trimesh_mesh)
