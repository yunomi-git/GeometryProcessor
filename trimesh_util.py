import trimesh
import numpy as np
from stopwatch import Stopwatch
from tqdm import tqdm
import util
import matplotlib.pyplot as plt


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
        self.facets = mesh.faces

        self.edges = mesh.edges
        self.num_edges = len(self.edges)

        self.vertices = mesh.vertices
        self.num_vertices = len(self.vertices)

    def sample_and_get_normals(self, count=50000):
        sample_points, face_index = trimesh.sample.sample_surface_even(mesh=self.mesh, count=count)
        normals = self.facet_normals[face_index]
        return sample_points, normals

    def calculate_overhangs_samples(self, cutoff_angle_rad=np.pi / 2.0, return_num_samples=False):
        layer_height = 0.4
        trimesh.repair.fix_normals(self.mesh, multibody=True)
        samples, normals = self.sample_and_get_normals()
        normals_z = normals[:, 2]
        sample_angles = np.arcsin(normals_z)  #arcsin calculates overhang angles as < 0
        samples_above_floor = samples[:, 2] > (layer_height + self.bound_lower[2])
        overhang_indices = np.logical_and(sample_angles > -np.pi / 2.0, sample_angles < -cutoff_angle_rad)
        overhang_indices = np.logical_and(overhang_indices, samples_above_floor)

        overhang_samples = samples[overhang_indices]
        overhang_angles = -sample_angles[overhang_indices]

        if return_num_samples:
            return overhang_samples, overhang_angles, len(samples)
        else:
            return overhang_samples, overhang_angles

    def calculate_stairstep_samples(self, cutoff_angle_rad=np.pi / 2.0):
        samples, normals = self.sample_and_get_normals()
        sample_z = normals[:, 2]
        sample_angles = np.arcsin(sample_z)  # overhang angles will be < 0
        stairstep_indices = np.logical_and(sample_angles < np.pi / 2.0 * 0.99, sample_angles > cutoff_angle_rad)

        stairstep_samples = samples[stairstep_indices]
        stairstep_angles = sample_angles[stairstep_indices]

        return stairstep_samples, stairstep_angles


    def calculate_thicknesses_samples(self, count=50000, return_num_samples=False):
        trimesh.repair.fix_normals(self.mesh, multibody=True)
        origins, normals = self.sample_and_get_normals(count)

        facet_offset = -normals * 0.001  # This offset needs to be tuned based on stl dimensions
        hits, ray_ids, tri_ids = self.mesh.ray.intersects_location(ray_origins=origins + facet_offset,
                                                                   ray_directions=-normals,
                                                                   multiple_hits=False)

        hit_origins = origins[ray_ids]
        print("hits", len(hit_origins))

        distances = np.linalg.norm(hits - hit_origins, axis=1)
        wall_thicknesses = distances
        if return_num_samples:
            return hit_origins, wall_thicknesses, len(tri_ids)
        else:
            return hit_origins, wall_thicknesses

    def calculate_gap_samples(self, count=50000, return_num_samples=False):
        trimesh.repair.fix_normals(self.mesh, multibody=True)
        origins, normals = self.sample_and_get_normals(count)

        facet_offset = normals * 0.1  # This offset needs to be tuned based on stl dimensions
        hits, ray_ids, tri_ids = self.mesh.ray.intersects_location(ray_origins=origins + facet_offset,
                                                                   ray_directions=normals,
                                                                   multiple_hits=False)
        hit_origins = origins[ray_ids]
        distances = np.linalg.norm(hits - hit_origins, axis=1)
        gap_sizes = distances

        if return_num_samples:
            return hit_origins, gap_sizes, len(tri_ids)
        else:
            return hit_origins, gap_sizes

    def get_vertices_of_facets(self, facet_indices):
        # Get list of faces
        facets = self.facets[facet_indices]
        # convert to 1D and remove duplicates
        vertices = set(facets.reshape(len(facets) * 3))
        return vertices

    def calculate_thicknesses_facets(self):
        trimesh.repair.fix_normals(self.mesh, multibody=True)

        num_facets = self.num_facets

        facet_offset = -self.facet_normals * 0.001
        hits, ray_ids, tri_ids = self.mesh.ray.intersects_location(ray_origins=self.facet_centroids + facet_offset,
                                                 ray_directions=-self.facet_normals,
                                                 multiple_hits=False)

        hit_origins = self.facet_centroids[tri_ids]
        distances = np.linalg.norm(hits - hit_origins, axis=1)
        wall_thicknesses = np.ones(num_facets) * NO_GAP_VALUE
        wall_thicknesses[tri_ids] = distances

        return wall_thicknesses

    def calculate_gap_facets(self):
        trimesh.repair.fix_normals(self.mesh, multibody=True)

        num_facets = self.num_facets

        facet_offset = self.facet_normals * 0.1
        hits, ray_ids, tri_ids = self.mesh.ray.intersects_location(ray_origins=self.facet_centroids + facet_offset,
                                                                   ray_directions=self.facet_normals,
                                                                   multiple_hits=False)
        hit_origins = self.facet_centroids[tri_ids]
        distances = np.linalg.norm(hits - hit_origins, axis=1)
        gap_sizes = np.ones(num_facets) * NO_GAP_VALUE
        gap_sizes[tri_ids] = distances
        return gap_sizes

def show_sampled_values(mesh, points, values, normalize=True):
    s = trimesh.Scene()

    if normalize:
        values = util.normalize_minmax_01(values)

    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    colors = 255.0 * cmap(values)
    colors[:, 3] = int(0.8 * 255)
    point_cloud = trimesh.points.PointCloud(vertices=points,
                                            colors=colors)
    s.add_geometry(point_cloud)
    s.add_geometry(mesh)
    s.show()

def show_mesh_with_facet_colors(mesh, values: np.ndarray, normalize=True):
    s = trimesh.Scene()

    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    empty_color = np.array([100, 100, 100, 255])

    if normalize:
        values[values != NO_GAP_VALUE] = util.normalize_minmax_01(values[values != NO_GAP_VALUE])
    mesh.visual.face_colors = cmap(values)
    mesh.visual.face_colors[values == NO_GAP_VALUE] = empty_color

    s.add_geometry(mesh)
    s.show()

def show_mesh(mesh):
    s = trimesh.Scene()
    s.add_geometry(mesh)
    s.show()

def show_mesh_with_orientation(mesh):
    mesh_aux = MeshAuxilliaryInfo(mesh)
    colors = util.direction_to_color(mesh_aux.facet_normals)
    mesh.visual.face_colors = colors
    s = trimesh.Scene()
    s.add_geometry(mesh)
    s.show()


if __name__=="__main__":
    print("hi")

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
        stopwatch.print_time()

        stopwatch.start()
        fill_orig = voxels.is_filled(random_point)
        print("orig")
        stopwatch.print_time()

        print("Equal?: ", fill_new == fill_orig)
        print("Point: ", random_point)
        print("Fill?: ", fill_new)
        print("------")

