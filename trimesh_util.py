import trimesh
import numpy as np
from stopwatch import Stopwatch
TRIMESH_TEST_MESH = trimesh.Trimesh(vertices=np.array([[0.0, 1, 0.0], [1, 0.0, 0.0], [0, 0, 0], [0.0, 0.01, 1]]),
                                    faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))

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