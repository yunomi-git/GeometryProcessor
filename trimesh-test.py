import numpy as np
import trimesh

if __name__=="__main__":
    # attach to logger so trimesh messages will be printed to console
    trimesh.util.attach_to_log()

    # some formats represent multiple meshes with multiple instances
    # the loader tries to return the datatype which makes the most sense
    # which will for scene-like files will return a `trimesh.Scene` object.
    # if you *always* want a straight `trimesh.Trimesh` you can ask the
    # loader to "force" the result into a mesh through concatenation
    mesh = trimesh.load('stls/low-res.stl', force='mesh')

    # is the current mesh watertight?
    mesh.is_watertight
    # what's the euler number for the mesh?
    mesh.euler_number

    # edges, vertices = trimesh.path.polygons.medial_axis(mesh)

    # since the mesh is watertight, it means there is a
    # volumetric center of mass which we can set as the origin for our mesh
    mesh.vertices -= mesh.center_mass

    # facets are groups of coplanar adjacent faces
    # colors are 8 bit RGBA by default (n, 4) np.uint8
    for facet in mesh.facets:
        mesh.visual.face_colors[facet] = trimesh.visual.random_color()

    # preview mesh in an opengl window if you installed pyglet and scipy with pip
    mesh.show()

    # transform method can be passed a (4, 4) matrix and will cleanly apply the transform
    mesh.apply_transform(trimesh.transformations.random_rotation_matrix())

    # axis aligned bounding box is available
    mesh.bounding_box.extents

    # a minimum volume oriented bounding box also available
    # primitives are subclasses of Trimesh objects which automatically generate
    # faces and vertices from data stored in the 'primitive' attribute
    mesh.bounding_box_oriented.primitive.extents
    mesh.bounding_box_oriented.primitive.transform

    # the bounding box is a trimesh.primitives.Box object, which subclasses
    # Trimesh and lazily evaluates to fill in vertices and faces when requested
    # (press w in viewer to see triangles)
    mesh.show()
    # (mesh + mesh.bounding_box_oriented).show()

