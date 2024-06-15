import pyvista as pv
import trimesh

import paths
import trimesh_util
import numpy as np
from pyvista import examples


def convert_to_pv_mesh(vertices, faces):
    pad = 3.0 * np.ones((mesh_aux.num_faces, 1))
    faces = np.concatenate((pad, faces), axis=1)
    faces = np.hstack(faces).astype(np.int64)
    mesh = pv.PolyData(vertices, faces)
    return mesh

if __name__=="__main__":
    pl = pv.Plotter(shape=(1, 2))
    stl_path = paths.HOME_PATH + "stls/Octocat.stl"
    mesh = trimesh.load(stl_path)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    mesh = convert_to_pv_mesh(mesh_aux.vertices, mesh_aux.faces)

    points, curvature = mesh_aux.calculate_curvature_at_points(origins=mesh_aux.vertices, face_ids=None, curvature_method="abs", return_num_samples=False)

    pl.add_mesh(mesh, show_edges=True, scalars=curvature, show_scalar_bar=True)
    pl.show()
    # points, curvature = mesh_aux.calculate_curvature_samples(count=4096)

    # pl.subplot(0, 0)
    # curvature = curvature - curvature.min(axis=0)
    # curvature /= curvature.max(axis=0)
    # actor = pl.add_points(
    #     points,
    #     scalars=curvature,
    #     render_points_as_spheres=True,
    #     point_size=10,
    #     show_scalar_bar=True,
    # )
    # pl.add_text('Curvature', color='w')
    # actor.mapper.lookup_table.cmap = 'jet'
    #
    # points, thickness = mesh_aux.calculate_thicknesses_samples(count=4096)
    # pl.subplot(0, 1)
    # thickness = thickness - thickness.min(axis=0)
    # thickness /= thickness.max(axis=0)
    # actor = pl.add_points(
    #     points,
    #     scalars=thickness,
    #     render_points_as_spheres=True,
    #     point_size=10,
    #     show_scalar_bar=True,
    # )
    # pl.add_text('Thickness', color='w')
    # actor.mapper.lookup_table.cmap = 'jet'
    #
    # pl.link_views()
    # pl.show()
