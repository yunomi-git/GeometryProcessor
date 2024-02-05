# https://devtalk.blender.org/t/alternative-in-2-80-to-create-meshes-from-python-using-the-tessfaces-api/7445/3
# Example of creating a polygonal mesh in Python from numpy arrays
# Note: this is Python 3.x code
#
# $ blender -P create_mesh.py
#
# See this link for more information on this part of the API:
# https://docs.blender.org/api/blender2.8/bpy.types.Mesh.html
#
# Paul Melis (paul.melis@surfsara.nl), SURFsara, 24-05-2019
import bpy
import numpy
import bmesh

# Note: we DELETE all objects in the scene and only then create the new mesh!
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Vertices and edges (straightforward)

vertices = numpy.array([
    0, 0, 0,
    2, 0, 0,
    2, 2, 0.2,
    0, 2, 0.2,
    1, 3, 1,
    1, -1, -1,
    0, -2, -1,
    2, -2, -1
], dtype=numpy.float32)

# Setting edges is optional, as they get created automatically for
# any provided polygons. However, if you need edges that exist separately
# from polygons then use this array.
# XXX these edges only seem to show up after going in-and-out of edit mode.
# edges = numpy.array([
#     5, 6,
#     6, 7,
#     5, 7
# ], dtype=numpy.int32)

num_vertices = vertices.shape[0] // 3
# num_edges = edges.shape[0] // 2

# Polygons are defined in loops. Here, we define one quad and two triangles

vertex_index = numpy.array([
    0, 1, 2, 3,
    4, 3, 2,
    0, 5, 1
], dtype=numpy.int32)

# For each polygon the start of its vertex indices in the vertex_index array
loop_start = numpy.array([
    0, 4, 7
], dtype=numpy.int32)

# Length of each polygon in number of vertices
loop_total = numpy.array([
    4, 3, 3
], dtype=numpy.int32)

num_vertex_indices = vertex_index.shape[0]
num_loops = loop_start.shape[0]

# Texture coordinates per vertex *per polygon loop*.
# uv_coordinates = numpy.array([
#     0, 0,
#     1, 0,
#     1, 1,
#     0, 1,
#
#     0.5, 1,
#     0, 0,
#     1, 0,
#
#     0, 1,
#     0.5, 0,
#     1, 1
# ], dtype=numpy.float32)
#
# # Vertex color per vertex *per polygon loop*
# vertex_colors = numpy.array([
#     1, 0, 0,
#     1, 0, 0,
#     1, 0, 0,
#     1, 0, 0,
#
#     0, 1, 0,
#     0, 1, 0,
#     0, 1, 0,
#
#     1, 0, 0,
#     0, 1, 0,
#     0, 0, 1
# ], dtype=numpy.float32)
#
# assert uv_coordinates.shape[0] == 2 * vertex_index.shape[0]
# assert vertex_colors.shape[0] == 3 * vertex_index.shape[0]

# Create mesh object based on the arrays above

mesh = bpy.data.meshes.new(name='created mesh')

mesh.vertices.add(num_vertices)
mesh.vertices.foreach_set("co", vertices)

# mesh.edges.add(num_edges)
# mesh.edges.foreach_set("vertices", edges)

mesh.loops.add(num_vertex_indices)
mesh.loops.foreach_set("vertex_index", vertex_index)

mesh.polygons.add(num_loops)
mesh.polygons.foreach_set("loop_start", loop_start)
mesh.polygons.foreach_set("loop_total", loop_total)

# # Create UV coordinate layer and set values
# uv_layer = mesh.uv_layers.new()
# for i, uv in enumerate(uv_layer.data):
#     uv.uv = uv_coordinates[2 * i:2 * i + 2]
#
# # Create vertex color layer and set values
# vcol_lay = mesh.vertex_colors.new()
# for i, col in enumerate(vcol_lay.data):
#     col.color[0] = vertex_colors[3 * i + 0]
#     col.color[1] = vertex_colors[3 * i + 1]
#     col.color[2] = vertex_colors[3 * i + 2]
#     col.color[3] = 1.0  # Alpha?

# We're done setting up the mesh values, update mesh object and
# let Blender do some checks on it
mesh.update()
mesh.validate()
print("done")

# # Create Object whose Object Data is our new mesh
# obj = bpy.data.objects.new('created object', mesh)
#
# # Add *Object* to the scene, not the mesh
# scene = bpy.context.scene
# scene.collection.objects.link(obj)
#
# # Select the new object and make it active
# bpy.ops.object.select_all(action='DESELECT')
# obj.select_set(True)
# bpy.context.view_layer.objects.active = obj