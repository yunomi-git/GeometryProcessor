# Why does ray trace fail for some thicknesses?

import trimesh
import numpy as np
import trimesh_util
import paths

def get_hit_for_facet(i, mesh, mesh_aux):
    hits = mesh.ray.intersects_location(ray_origins=mesh_aux.face_centroids[np.newaxis, i, :],
                                        ray_directions=-mesh_aux.face_normals[np.newaxis, i, :],
                                        multiple_hits=True)[0]
    # first_hit = hits[0]
    # start = mesh_aux.facet_centroids[i, :]
    # end = first_hit
    # return start, end
    if np.isclose(hits[0, :], mesh_aux.face_centroids[i, :]).all():
        hits = hits[1:, :]
    return hits

def check_ray_for_mesh(mesh):
    mesh = trimesh_util.get_largest_submesh(mesh)


    if not (trimesh_util.mesh_is_valid(mesh) and mesh.is_watertight):
        return

    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    num_samples = 10000
    origins, normals = mesh_aux.sample_and_get_normals(num_samples)
    min_bound = min(mesh_aux.bound_length)
    normal_scale = 5e-5 * min_bound

    facet_offset = -normals * normal_scale  # This offset needs to be tuned based on stl dimensions
    hits, ray_ids, tri_ids = mesh_aux.mesh.ray.intersects_location(ray_origins=origins + facet_offset,
                                                                   ray_directions=-normals,
                                                                   multiple_hits=False)

    # Find out where ray trace failed
    missed_ids = [x for x in range(num_samples) if x not in ray_ids]
    if len(missed_ids) > 0:
        print(missed_ids)
    missed_origins = origins[missed_ids]
    missed_rays = -normals[missed_ids]

    for i in range(len(missed_origins)):
        s = trimesh.Scene()

        # hits = get_hit_for_facet(i, mesh, mesh_aux)
        start = missed_origins[i]
        end = start + missed_rays[i]

        # distances = np.linalg.norm(start - end)

        # Create the line
        line = trimesh.load_path(np.array([start.tolist(), end.tolist()]))
        s.add_geometry(line)

        # Create the start and end points
        main_color = np.array([0, 255, 0, 100])
        hit_color = np.array([255, 0, 0, 150])

        # colors_list = np.stack([main_color,
        #                                                    np.repeat(hit_color, len(hits), axis=0)])
        colors = np.concatenate((main_color[np.newaxis, :]), axis=0)
        points = trimesh.points.PointCloud(vertices=np.concatenate([start[np.newaxis, :]], axis=0),
                                           colors=np.concatenate([main_color[np.newaxis, :]],
                                                                 axis=0))

        mesh.visual.face_colors = np.array([100, 100, 100, 255])

        s.add_geometry(mesh)
        s.add_geometry(points)
        s.show()

if __name__=="__main__":
    # mesh_path = paths.get_thingiverse_stl_path(92)
    # mesh = trimesh.load(mesh_path)
    # check_ray_for_mesh(mesh)

    for i in range(5000):
        mesh_path = paths.get_thingiverse_stl_path(i)
        mesh = trimesh.load(mesh_path)
        check_ray_for_mesh(mesh)

