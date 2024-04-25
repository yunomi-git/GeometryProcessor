import pyvista as pv
import trimesh

import paths
import trimesh_util

if __name__=="__main__":
    pl = pv.Plotter(shape=(1, 2))
    stl_path = paths.HOME_PATH + "stls/low-res.stl"
    mesh = trimesh.load(stl_path)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    points, curvature = mesh_aux.calculate_curvature_samples(count=4096)
    pl.subplot(0, 0)
    curvature = curvature - curvature.min(axis=0)
    curvature /= curvature.max(axis=0)
    actor = pl.add_points(
        points,
        scalars=curvature,
        render_points_as_spheres=True,
        point_size=10,
        show_scalar_bar=True,
    )
    pl.add_text('Curvature', color='w')
    actor.mapper.lookup_table.cmap = 'jet'

    points, thickness = mesh_aux.calculate_thicknesses_samples(count=4096)
    pl.subplot(0, 1)
    thickness = thickness - thickness.min(axis=0)
    thickness /= thickness.max(axis=0)
    actor = pl.add_points(
        points,
        scalars=thickness,
        render_points_as_spheres=True,
        point_size=10,
        show_scalar_bar=True,
    )
    pl.add_text('Thickness', color='w')
    actor.mapper.lookup_table.cmap = 'jet'

    pl.link_views()
    pl.show()
