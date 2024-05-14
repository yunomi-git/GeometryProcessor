import trimesh
import numpy as np
import trimesh_util
import random
import paths

def get_hit_for_facet(i, mesh, mesh_aux):
    normal = mesh_aux.face_normals[i, :]
    origin = mesh_aux.face_centroids[i, :]
    facet_offset = normal * 0.001
    hits = mesh.ray.intersects_location(ray_origins=(origin + facet_offset)[np.newaxis, :],
                                        ray_directions=normal[np.newaxis, :],
                                        multiple_hits=True)[0]

    return hits

if __name__=="__main__":
    mesh_path = paths.get_onshape_stl_path(2)
    # mesh_path = 'stls/Antenna_DJI.stl'
    mesh = trimesh.load(mesh_path)
    # mesh = trimesh_util.TRIMESH_TEST_MESH

    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    num_facets = mesh_aux.num_faces

    for j in range(num_facets):
        i = random.randint(0, num_facets)
        s = trimesh.Scene()

        hits = get_hit_for_facet(i, mesh, mesh_aux)
        normal = mesh_aux.face_normals[i, :]
        start = mesh_aux.face_centroids[i, :]
        # distances = np.linalg.norm(start - end)

        # Create the line
        for end in hits:
            line = trimesh.load_path(np.array([start.tolist(), end.tolist()]))
            s.add_geometry(line)

        # Create normal direction
        line = trimesh.load_path(np.array([start.tolist(), (start + normal * 1).tolist()]))
        line.colors = np.array([[0, 0, 255, 100]])
        s.add_geometry(line)

        # Create the start and end points
        main_color = np.array([0, 255, 0, 100])
        hit_color = np.array([255, 0, 0, 150])

        colors = np.concatenate((main_color[np.newaxis, :], np.repeat(hit_color[np.newaxis, :], len(hits), axis=0)), axis=0)
        points = trimesh.points.PointCloud(vertices=np.concatenate([start[np.newaxis, :], hits], axis=0),
                                           colors=np.concatenate([main_color[np.newaxis, :],
                                                                  np.repeat(hit_color[np.newaxis, :], len(hits), axis=0)],
                                                                 axis=0))

        mesh.visual.face_colors = np.array([100, 100, 100, 40])

        s.add_geometry(mesh)
        s.add_geometry(points)
        s.show()