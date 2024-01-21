import trimesh
import numpy as np
from stopwatch import Stopwatch
from tqdm import tqdm


TRIMESH_TEST_MESH = trimesh.Trimesh(vertices=np.array([[0.0, 1, 0.0], [1, 0.0, 0.0], [0, 0, 0], [0.0, 0.01, 1]]),
                                    faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))
NO_GAP_VALUE = -1

class MeshAuxilliaryInfo:
    def __init__(self, mesh):
        self.mesh = mesh
        self.bound_lower = mesh.bounds[0, :].copy()
        self.bound_upper = mesh.bounds[1, :].copy()
        self.bound_length = self.bound_upper - self.bound_lower

        self.facet_centroids = mesh.triangles_center
        self.facet_normals = mesh.face_normals
        self.facet_areas = mesh.area_faces
        self.num_facets = len(self.facet_centroids)

    def get_thicknesses(self):
        trimesh.repair.fix_normals(self.mesh, multibody=True)

        num_facets = self.num_facets

        facet_offset = -self.facet_normals * 0.001
        hits = self.mesh.ray.intersects_location(ray_origins=self.facet_centroids + facet_offset,
                                                 ray_directions=-self.facet_normals,
                                                 multiple_hits=False)[0]

        if len(hits) != num_facets:
            print("Trimesh thickness error: ", len(hits), " hits detected. ", num_facets, "hits expected.")
            return
        wall_thicknesses = np.linalg.norm(hits - self.facet_centroids, axis=1)
        return wall_thicknesses

    def calculate_and_show_gap(self):
        trimesh.repair.fix_normals(self.mesh, multibody=True)

        num_facets = self.num_facets
        gap_sizes = np.empty(num_facets)

        for i in tqdm(range(num_facets)):
            normal = self.facet_normals[i, :]
            origin = self.facet_centroids[i, :]
            facet_offset = normal * 0.001
            hits = self.mesh.ray.intersects_location(ray_origins=(origin + facet_offset)[np.newaxis, :],
                                                     ray_directions=normal[np.newaxis, :],
                                                     multiple_hits=False)[0]
            if len(hits) == 0:
                gap_sizes[i] = NO_GAP_VALUE
            else:
                first_hit = hits[0]
                distance = np.linalg.norm(origin - first_hit)
                gap_sizes[i] = distance
        return gap_sizes


### Below is voxel stuff. Unused.

def voxelize(mesh):
    bounds = mesh.bounds
    size = bounds[1, :] - bounds[0, :]

    nominal_mesh_size = 1.0
    nominal_voxel_size = 0.002
    min_scale = max(size / nominal_mesh_size)
    desired_voxel_size = min_scale * nominal_voxel_size

    # start = time.time()
    angel_voxel = mesh.voxelized(pitch=desired_voxel_size, method="ray")  # ray, subdivide, binvox
    # angel_voxel.fill(method='base')
    # base=fill_base,
    # orthographic=fill_orthographic,
    # holes=fill_holes,
    # print(time.time() - start)
    # print("---")

    return angel_voxel
    # return angel_voxel.as_boxes()

class VoxelAuxilliaryInfo:
    def __init__(self, voxel):
        self.voxel = voxel
        self.bound_lower = voxel.bounds[0, :].copy()
        self.bound_upper = voxel.bounds[1, :].copy()
        self.bound_length = self.bound_upper - self.bound_lower
        num_grids = np.array(voxel.shape)
        self.grid_size = np.divide(self.bound_length, num_grids)
        self.bound_lower += self.grid_size / 2.0
        self.bound_upper += self.grid_size / 2.0

    def check_voxel_is_filled(self, point):
        grid_index = np.floor((point - self.bound_lower) / self.grid_size).astype(int)
        return self.voxel.encoding.dense[grid_index[0], grid_index[1], grid_index[2]]

def check_voxel_fill_equivalency():
    stopwatch = Stopwatch()
    # mesh_path = 'stls/low-res.stl'
    # mesh = trimesh.load(mesh_path, force="mesh")
    mesh = TRIMESH_TEST_MESH

    voxels = voxelize(mesh)
    voxel_auxiliary = VoxelAuxilliaryInfo(voxels)

    s = trimesh.Scene()
    # s.add_geometry(mesh)
    s.add_geometry(voxels.as_boxes(colors=np.array([200, 50, 50, 150])))
    s.show()

    for i in range(10):
        # random_point = np.zeros(3)
        random_point = np.random.rand(3) * voxel_auxiliary.bound_length + voxel_auxiliary.bound_lower
        stopwatch.start()
        fill_new = voxel_auxiliary.check_voxel_is_filled(random_point)
        print("new")
        stopwatch.get_time()

        stopwatch.start()
        fill_orig = voxels.is_filled(random_point)
        print("orig")
        stopwatch.get_time()

        print("Equal?: ", fill_new == fill_orig)
        print("Point: ", random_point)
        print("Fill?: ", fill_new)
        print("------")

if __name__=="__main__":
    check_voxel_fill_equivalency()