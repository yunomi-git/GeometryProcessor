# Why does ray trace fail for some thicknesses?

import trimesh
import numpy as np
import trimesh_util
import paths
import dataset.FolderManager as fm

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

def get_hits(mesh, sampling_method="vertices", multiple_hits=False):
    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    if sampling_method == "vertices":
        origins = mesh_aux.vertices
        normals = mesh_aux.vertex_normals
    elif sampling_method == "faces":
        origins = mesh_aux.face_centroids
        normals = mesh_aux.face_normals
    else:
        num_samples = 10000
        origins, normals = mesh_aux.sample_and_get_normals(num_samples)
    min_bound = min(mesh_aux.bound_length)
    normal_scale = 5e-4 * min_bound

    facet_offset = -normals * normal_scale  # This offset needs to be tuned based on stl dimensions
    hits, ray_ids, tri_ids = mesh_aux.mesh.ray.intersects_location(ray_origins=origins + facet_offset,
                                                                   ray_directions=-normals,
                                                                   multiple_hits=multiple_hits)
    return hits, ray_ids, tri_ids, origins, normals, facet_offset


def count_valid_meshes(mesh, sampling_method="vertices"):
    mesh = trimesh_util.get_largest_submesh(mesh)

    if not (trimesh_util.mesh_is_valid(mesh) and mesh.is_watertight):
        return None

    hits, ray_ids, tri_ids, origins, normals, facet_offset = get_hits(mesh, sampling_method=sampling_method, multiple_hits=False)

    # Find out where ray trace failed
    missed_ids = [x for x in range(len(origins)) if x not in ray_ids]
    if len(missed_ids) > 0:
        print("invalid")
    else:
        print("valid")


def check_ray_for_mesh(mesh, sampling_method="vertices"):
    mesh = trimesh_util.get_largest_submesh(mesh)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    min_bound = np.min(mesh_aux.bound_length)

    if not (trimesh_util.mesh_is_valid(mesh) and mesh.is_watertight):
        return None

    hits, ray_ids, tri_ids, origins, normals, facet_offset = get_hits(mesh, sampling_method=sampling_method, multiple_hits=False)

    # Find out where ray trace failed
    missed_ids = [x for x in range(len(origins)) if x not in ray_ids]
    if len(missed_ids) > 0:
        print(missed_ids)
        missed_origins = origins[missed_ids]
        missed_rays = -normals[missed_ids]
        # hits = get_hit_for_facet(i, mesh, mesh_aux)

        s = trimesh.Scene()
        for i in range(len(missed_origins)):
            start = missed_origins[i]
            end = start + missed_rays[i] * min_bound * 0.25
            if np.linalg.norm(missed_rays[i]) < 1e-15:
                print("0 norm normal")
                continue

            # Create the line
            line = trimesh.load_path(np.array([start.tolist(), end.tolist()]))
            s.add_geometry(line)

        # Create the start and end points
        main_color = np.array([0, 255, 0, 100])
        origin_point = trimesh.points.PointCloud(vertices=missed_origins, colors=main_color)
        # intersect_points = trimesh.points.PointCloud(vertices=np.concatenate([start[np.newaxis, :]], axis=0),
        #                                    colors=hit_color)

        mesh.visual.face_colors = np.array([100, 100, 100, 100])

        s.add_geometry(mesh)
        s.add_geometry(origin_point)
        s.show()
    else:
        print("valid")

def show_ray_origins(mesh,
                            sampling_method="vertices" #vertices or sample
                            ):
    mesh = trimesh_util.get_largest_submesh(mesh)
    if not (trimesh_util.mesh_is_valid(mesh) and mesh.is_watertight):
        return None

    hits, ray_ids, tri_ids, origins, normals, facet_offset = get_hits(mesh, sampling_method=sampling_method, multiple_hits=False)

    # Find out where ray trace failed
    missed_ids = [x for x in range(len(origins)) if x not in ray_ids]
    if len(missed_ids) > 0:
        print(len(missed_ids))
        missed_origins = origins[missed_ids]
        hit_origins = origins[ray_ids]

        s = trimesh.Scene()

        hit_color = np.array([0, 255, 0, 255])
        miss_color = np.array([255, 0, 0, 255])

        points_hit = trimesh.points.PointCloud(vertices=hit_origins,
                                           colors=hit_color)

        points_missed = trimesh.points.PointCloud(vertices=missed_origins,
                                           colors=miss_color)

        mesh.visual.face_colors = np.array([100, 100, 100, 250])

        s.add_geometry(mesh)
        s.add_geometry(points_hit)
        s.add_geometry(points_missed)

        s.show()

def show_failed_thicknesses(mesh,
                            sampling_method="vertices" #vertices or sample
                            ):
    mesh = trimesh_util.get_largest_submesh(mesh)
    if not (trimesh_util.mesh_is_valid(mesh) and mesh.is_watertight):
        return None

    hits, ray_ids, tri_ids, origins, normals, facet_offset = get_hits(mesh, sampling_method=sampling_method, multiple_hits=False)

    # Find out where ray trace failed
    missed_ids = [x for x in range(len(origins)) if x not in ray_ids]
    if len(missed_ids) > 0:
        print(len(missed_ids))
        missed_origins = origins[missed_ids]
        missed_offset = facet_offset[missed_ids]

        s = trimesh.Scene()

        green = np.array([0, 255, 0, 255])
        red = np.array([255, 0, 0, 255])

        points = trimesh.points.PointCloud(vertices=missed_origins,
                                           colors=green)
        offset_points = trimesh.points.PointCloud(vertices=missed_origins + missed_offset,
                                           colors=red)

        mesh.visual.face_colors = np.array([100, 100, 100, 100])

        s.add_geometry(mesh)
        s.add_geometry(points)
        s.add_geometry(offset_points)

        s.show()



if __name__=="__main__":
    # mesh_path = paths.get_thingiverse_stl_path(92)
    # mesh = trimesh.load(mesh_path)
    # check_ray_for_mesh(mesh)
    folder_manager = fm.DirectoryPathManager(base_path=paths.RAW_DATASETS_PATH + "Thingi10k_Remesh_Normalized/", base_unit_is_file=True)
    files = folder_manager.get_files_absolute()[:50]

    for file_path in files:
        # mesh_path = paths.get_thingiverse_stl_path(i)
        mesh = trimesh.load(file_path)
        count_valid_meshes(mesh, sampling_method="vertices")

