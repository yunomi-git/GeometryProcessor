import numpy as np
import pyvista as pv
import trimesh_util
import util

def convert_to_pv_mesh(vertices, faces):
    pad = 3.0 * np.ones((len(faces), 1))
    faces = np.concatenate((pad, faces), axis=1)
    faces = np.hstack(faces).astype(np.int64)
    mesh = pv.PolyData(vertices, faces)
    return mesh

def show_mesh(vertices, faces):
    mesh = convert_to_pv_mesh(vertices, faces)

    default_size = 600
    pl = pv.Plotter(window_size=[default_size, default_size])
    # pl.subplot(0, 0)
    actor1 = pl.add_mesh(
        mesh,
        show_edges=True,
        # scalars=actual.flatten(),
        # show_scalar_bar=True,
        # scalar_bar_args={'title': 'Actual',
        #                  'n_labels': 3},
        # clim=[min_value, max_value]
    )
    # pl.add_text('Actual', color='black')
    # actor1.mapper.lookup_table.cmap = 'jet'

    pl.show()

def show_meshes_in_grid(meshes, r=4, c=4):
    num_data = len(meshes)
    num_visualized = c * r
    num_iterations = int(np.ceil(num_data / num_visualized))

    for i in range(num_iterations):
        pl = pv.Plotter(shape=(r, c))
        for ri in range(r):
            for ci in range(c):
                idx = i * r * c + ri * c + ci
                if idx >= num_data:
                    break
                mesh_vertices = meshes[idx].vertices
                mesh_faces = meshes[idx].faces
                # mesh_labels = labels[idx]
                mesh = convert_to_pv_mesh(mesh_vertices, mesh_faces)
                pl.subplot(ri, ci)
                actor = pl.add_mesh(
                    mesh,
                    # scalars=mesh_labels,
                    # render_points_as_spheres=True,
                    # point_size=5,
                    # rgb=True,
                    # show_scalar_bar=True,
                    # text="Curvature"
                )
                # actor.mapper.lookup_table.cmap = 'jet'
                pl.show_bounds(grid=True, all_edges=False,  font_size=10)

        pl.link_views()
        pl.show()

def show_mesh_z_mag(trimesh_mesh):
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(trimesh_mesh)

    def get_angle_magnitudes(direction):
        horiz_dir = np.sqrt(direction[:, 1]**2 + direction[:, 0] ** 2)
        vert_dir = direction[:, 2]
        pitch = np.arctan2(vert_dir, horiz_dir)  # range -pi/2, pi/2
        pitch = np.abs(pitch) / (np.pi / 2) # range is 0, pi/2. change to 0, 1
        return pitch

    angle_magnitudes = get_angle_magnitudes(mesh_aux.face_normals)

    mesh = convert_to_pv_mesh(mesh_aux.vertices, mesh_aux.faces)

    default_size = 600
    pl = pv.Plotter(window_size=[default_size, default_size])
    actor1 = pl.add_mesh(
        mesh,
        show_edges=True,
        scalars=angle_magnitudes.flatten(),
        show_scalar_bar=False,
        # scalar_bar_args={'title': 'Actual',
        #                  'n_labels': 3},
        # clim=[min_value, max_value]
    )
    # pl.add_text('Actual', color='black')
    pl.show_bounds()
    actor1.mapper.lookup_table.cmap = 'jet'

    pl.show()
