import bpy
import os
import trimesh
import bmesh
import paths
import numpy as np
import trimesh_util
import util


def convert_blender_to_trimesh(blender_mesh):
    num_vert = len(blender_mesh.vertices)
    verts = np.empty(num_vert * 3, dtype=np.float64)
    blender_mesh.vertices.foreach_get('co', verts)
    verts.shape = (num_vert, 3)

    num_facet = len(blender_mesh.polygons)
    facets = np.empty(num_facet * 3, dtype=np.int32)
    blender_mesh.polygons.foreach_get('vertices', facets)
    facets.shape = (num_facet, 3)

    trimesh_mesh = trimesh.Trimesh(vertices=verts,
                                   faces=facets)
    return trimesh_mesh

def convert_trimesh_to_blender(trimesh_mesh, name="a"):
    trimesh_aux = trimesh_util.MeshAuxilliaryInfo(trimesh_mesh)

    # Vertices
    num_vertices = trimesh_aux.num_vertices
    # num_edges = trimesh_aux.num_edges
    num_facets = trimesh_aux.num_facets

    vertices = np.array(trimesh_aux.vertices).reshape(num_vertices * 3).astype(np.float32)
    # edges = trimesh_aux.edges.reshape(num_edges * 2).astype(np.int32)
    vertex_index = np.array(trimesh_aux.facets).reshape(num_facets * 3).astype(np.int32)

    # For each polygon the start of its vertex indices in the vertex_index array
    loop_start = np.arange(0, num_facets).astype(np.int32) * 3
    # # Length of each polygon in number of vertices
    loop_total = np.ones(num_facets).astype(np.int32) * 3

    # Create mesh object based on the arrays above
    mesh = bpy.data.meshes.new(name=name)

    mesh.vertices.add(num_vertices)
    mesh.vertices.foreach_set("co", vertices)

    # mesh.edges.add(num_edges)
    # mesh.edges.foreach_set("vertices", edges)

    mesh.loops.add(num_facets * 3)
    mesh.loops.foreach_set("vertex_index", vertex_index)

    mesh.polygons.add(num_facets)
    mesh.polygons.foreach_set("loop_start", loop_start)
    mesh.polygons.foreach_set("loop_total", loop_total)

    # update mesh object and let Blender do some checks on it
    mesh.update(calc_edges=True)
    mesh.validate()

    return mesh

class BlenderTrimeshManager:
    def __init__(self, trimesh_mesh):
        self.trimesh_mesh = trimesh_mesh
        self.blender_storage_mesh = convert_trimesh_to_blender(trimesh_mesh)
        self.blender_mesh = bmesh.new()
        self.blender_mesh.from_mesh(self.blender_storage_mesh)

    def update_trimesh_from_blender(self):
        self.blender_mesh.to_mesh(self.blender_storage_mesh)
        self.trimesh_mesh = convert_blender_to_trimesh(self.blender_storage_mesh)

    def update_blender_from_trimesh(self):
        self.blender_storage_mesh = convert_trimesh_to_blender(trimesh_mesh)
        self.blender_mesh.from_mesh(self.blender_storage_mesh)

    def get_blender_verts_from_indices(self, indices):
        self.blender_mesh.verts.ensure_lookup_table()
        return [self.blender_mesh.verts[i] for i in indices]

# def test_indices():
#     # Method: find highest point in trimesh. Print it. print corresponding index in blender mesh. values should be same
#
#     name = str(222)
#     mesh_path = paths.get_thingiverse_stl_path(name, get_by_order=False)
#     bpy.ops.import_mesh.stl(filepath=mesh_path)
#     blender_mesh = bpy.data.objects[name].data
#     # blender_mesh.update()
#
#     trimesh_mesh = convert_blender_to_trimesh(blender_mesh)
#
#     # Verify that they are the same
#     tri_vertices = trimesh_mesh.vertices
#     max_z_index = np.argmax(tri_vertices[:, 2])
#     max_z = tri_vertices[max_z_index, :]
#     print(max_z_index, max_z)
#
#     for vertex in blender_mesh.vertices:
#         if vertex.index == max_z_index:
#             print(vertex.index, vertex.co)


if __name__=="__main__":
    # Load trimesh
    # Find vertices with small cracks
    # Convert to blender
    # Do blender process
    # Convert back to trimesh
    # Plot

    name = str(222)
    mesh_path = paths.get_thingiverse_stl_path(name, get_by_order=False)
    trimesh_mesh = trimesh.load(mesh_path)
    trimesh_aux = trimesh_util.MeshAuxilliaryInfo(trimesh_mesh)

    thicknesses = trimesh_aux.calculate_thicknesses_facets()
    characteristic_length = np.mean(trimesh_aux.bound_length) / 20.0
    thin_facet_indices = util.get_indices_of_conditional(np.logical_and.reduce([thicknesses < characteristic_length, thicknesses != trimesh_util.NO_GAP_VALUE]))
    thicknesses[thicknesses > characteristic_length] = trimesh_util.NO_GAP_VALUE
    # trimesh_util.show_mesh_with_facet_colors(trimesh_mesh, values=thicknesses, normalize=True)
    vertices = trimesh_aux.get_vertices_of_facets(thin_facet_indices)

    # Do sampling to demonstrate thin sections
    points, values = trimesh_aux.calculate_thicknesses_samples()
    points[values > characteristic_length] = trimesh_util.NO_GAP_VALUE
    trimesh_util.show_sampled_values(trimesh_mesh, points, values)

    # trimesh_util.show_mesh(trimesh_mesh)
    mesh_manager = BlenderTrimeshManager(trimesh_mesh)

    # Grab vertices in bmesh mode
    blender_verts = mesh_manager.get_blender_verts_from_indices(vertices)
    for vert in blender_verts:
        # print(vert.normal)
        bmesh.ops.translate(mesh_manager.blender_mesh, verts=[vert],
                            vec=vert.normal * characteristic_length / 3.0)
    # print(thin_facet_indices)
    # bmesh.ops.translate(mesh_manager.blender_mesh, verts=blender_verts, vec=(characteristic_length, characteristic_length, 0))

    bmesh.ops.smooth_vert(mesh_manager.blender_mesh, verts=mesh_manager.blender_mesh.verts, factor=0.25,
                          use_axis_x=True, use_axis_y=True, use_axis_z=True)
    # bmesh.ops.rotate(
    #     mesh_manager.blender_mesh,
    #     verts=mesh_manager.blender_mesh.verts,
    #     cent=(0.0, 1.0, 0.0),
    #     matrix=mathutils.Matrix.Rotation(math.radians(50.0), 3, 'X'))
    mesh_manager.update_trimesh_from_blender()

    trimesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh_manager.trimesh_mesh)
    points, values = trimesh_aux.calculate_thicknesses_samples()
    points[values > characteristic_length] = trimesh_util.NO_GAP_VALUE
    trimesh_util.show_sampled_values(mesh_manager.trimesh_mesh, points, values)
    # trimesh_util.show_mesh(mesh_manager.trimesh_mesh)