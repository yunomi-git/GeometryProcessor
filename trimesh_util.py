import trimesh
import numpy as np
from util import Stopwatch
from tqdm import tqdm
import util
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import warnings

TRIMESH_TEST_MESH = trimesh.Trimesh(vertices=np.array([[0.0, 1, 0.0], [1, 0.0, 0.0], [0, 0, 0], [0.0, 0.01, 1]]),
                                    faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))
NO_GAP_VALUE = -1

def mesh_is_valid(mesh):
    if not isinstance(mesh, trimesh.Trimesh):
        return False

    if mesh.bounds is None:
        return False

    return True

class MeshAuxilliaryInfo:
    def __init__(self, mesh):
        self.is_valid = mesh_is_valid(mesh)
        if not self.is_valid:
            return

        # try:
        trimesh.repair.fix_normals(mesh, multibody=True)
        # except RuntimeError:
        #     self.is_valid = False
        #     return

        self.mesh = mesh

        self.bound_lower = mesh.bounds[0, :].copy()
        self.bound_upper = mesh.bounds[1, :].copy()
        self.bound_length = self.bound_upper - self.bound_lower

        self.face_centroids = mesh.triangles_center
        self.face_normals = mesh.face_normals
        self.face_areas = mesh.area_faces
        self.surface_area = mesh.area
        self.num_faces = len(self.face_centroids)
        self.faces = mesh.faces
        self.facet_defects = None

        self.edges = mesh.edges
        self.num_edges = len(self.edges)

        self.vertices = mesh.vertices
        self.num_vertices = len(self.vertices)
        self.vertex_normals = mesh.vertex_normals

    def get_vertices_and_normals(self):
        return self.vertices, self.vertex_normals

    def get_centroids_and_normals(self, return_face_ids=False):
        return self.face_centroids, self.face_normals

    def sample_and_get_normals(self, count=5000, use_weight="even", return_face_ids=False):
        ## even, curvature, face_area

        if use_weight == "even":
            sample_points, face_index = trimesh.sample.sample_surface_even(mesh=self.mesh, count=count)
            if len(sample_points) < count:
                remaining_count = count - len(sample_points)
                extra_points, extra_face_index = trimesh.sample.sample_surface(mesh=self.mesh, count=remaining_count)
                sample_points = np.concatenate((sample_points, extra_points))
                face_index = np.concatenate((face_index, extra_face_index))
        elif use_weight == "curve":
            surface_defects = self.calculate_surface_defects_facets(use_abs=True)
            log_defects = np.log(np.abs(surface_defects))
            log_defects -= np.min(log_defects)
            # log_defects[log_defects < 0] = 0
            log_defects = log_defects**3
            # log_defects = 1 / (1 + np.exp(-log_defects))
            # Convert to probabilities
            sample_points, face_index = trimesh.sample.sample_surface(mesh=self.mesh, count=count, face_weight=log_defects)
        elif use_weight == "mixed":
            ratio = 0.3
            even_count = int(count * ratio)
            even_sample_points, even_face_index = trimesh.sample.sample_surface_even(mesh=self.mesh, count=even_count)
            even_count = len(even_sample_points)

            surface_defects = self.calculate_surface_defects_facets(use_abs=True)
            log_defects = np.log(np.abs(surface_defects) + 1e-5) * self.face_areas
            if np.isnan(log_defects).any():
                print("NAN encountered")
            # Convert to probabilities
            log_defects -= np.mean(log_defects)
            log_defects[log_defects < 0] = 0
            log_defects = log_defects**3
            # log_defects = 1 / (1 + np.exp(-log_defects))
            curve_sample_points, curve_face_index = trimesh.sample.sample_surface(mesh=self.mesh,
                                                                                  count=count - even_count,
                                                                                  face_weight=log_defects)

            sample_points = np.concatenate((even_sample_points, curve_sample_points))
            face_index = np.concatenate((even_face_index, curve_face_index))
        else: # use_weight == "face_area":
            sample_points, face_index = trimesh.sample.sample_surface(mesh=self.mesh, count=count)

        normals = self.face_normals[face_index]

        if return_face_ids:
            return sample_points, normals, face_index
        else:
            return sample_points, normals

    def calculate_overhangs_samples(self, cutoff_angle_rad=np.pi / 4.0,
                                    layer_height=0.2,
                                    return_num_samples=False):
        trimesh.repair.fix_normals(self.mesh, multibody=True)
        samples, normals = self.sample_and_get_normals()
        normals_z = normals[:, 2]
        sample_angles = np.arcsin(normals_z)  #arcsin calculates overhang angles as < 0
        samples_above_floor = samples[:, 2] >= (layer_height + self.bound_lower[2])
        overhang_indices = np.logical_and(sample_angles > -np.pi / 2.0, sample_angles < -cutoff_angle_rad)
        overhang_indices = np.logical_and(overhang_indices, samples_above_floor)

        overhang_samples = samples[overhang_indices]
        overhang_angles = -sample_angles[overhang_indices]

        if return_num_samples:
            return overhang_samples, overhang_angles, len(samples)
        else:
            return overhang_samples, overhang_angles

    def calculate_stairstep_samples(self, min_angle_rad=np.pi / 4.0, max_angle_rad=np.pi / 2.0 * 0.95, return_num_samples=False):
        samples, normals = self.sample_and_get_normals()
        sample_z = normals[:, 2]
        sample_angles = np.arcsin(sample_z)  # overhang angles will be < 0
        stairstep_indices = np.logical_and(sample_angles < max_angle_rad, sample_angles > min_angle_rad)

        stairstep_samples = samples[stairstep_indices]
        stairstep_angles = sample_angles[stairstep_indices]

        if return_num_samples:
            return stairstep_samples, stairstep_angles, len(samples)
        else:
            return stairstep_samples, stairstep_angles


    def calculate_thicknesses_samples(self, count=50000, return_num_samples=False):
        trimesh.repair.fix_normals(self.mesh, multibody=True)
        origins, normals = self.sample_and_get_normals(count)

        return self.calculate_thickness_at_points(points=origins, normals=normals, return_num_samples=return_num_samples)


    def calculate_thickness_at_points(self, points, normals, return_num_samples=True):
        min_bound = min(self.bound_length)
        normal_scale = 5e-5 * min_bound
        facet_offset = -normals * normal_scale  # This offset needs to be tuned based on stl dimensions
        hits, ray_ids, tri_ids = self.mesh.ray.intersects_location(ray_origins=points + facet_offset,
                                                                   ray_directions=-normals,
                                                                   multiple_hits=False)
        hit_origins = points[ray_ids]

        distances = np.linalg.norm(hits - hit_origins, axis=1)
        wall_thicknesses = distances
        if return_num_samples:
            return hit_origins, wall_thicknesses, len(tri_ids)
        else:
            return hit_origins, wall_thicknesses

    def calculate_gap_samples(self, count=50000, return_num_samples=False):
        trimesh.repair.fix_normals(self.mesh, multibody=True)
        origins, normals = self.sample_and_get_normals(count)

        min_bound = min(self.bound_length)
        normal_scale = 5e-2 * min_bound
        facet_offset = normals * normal_scale  # 0.1 prev
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

    def calculate_curvature_samples(self, curvature_method="defect", count=50000, return_num_samples=False, sampling_method="even"):
        ## Methods: gaussian, mean, face
        origins, _, face_ids = self.sample_and_get_normals(count, use_weight=sampling_method, return_face_ids=True)
        return self.calculate_curvature_at_points(origins=origins, face_ids=face_ids,
                                                  curvature_method=curvature_method,
                                                  return_num_samples=return_num_samples)

    def calculate_curvature_at_points(self, origins, face_ids, curvature_method="abs", return_num_samples=False, use_abs=True):
        avg_facet_radii = np.sqrt(np.mean(self.face_areas)) / np.pi
        if curvature_method== "gaussian":
            curvatures = trimesh.curvature.discrete_gaussian_curvature_measure(self.mesh, origins, radius=avg_facet_radii * 20)
        elif curvature_method== "mean":
            curvatures = trimesh.curvature.discrete_mean_curvature_measure(self.mesh, origins, radius=avg_facet_radii * 20)
        elif curvature_method == "abs":
            radius = avg_facet_radii * 10
            g = trimesh.curvature.discrete_gaussian_curvature_measure(self.mesh, origins, radius=radius)
            m = trimesh.curvature.discrete_mean_curvature_measure(self.mesh, origins, radius=radius)
            curvatures = np.abs(g) + np.abs(m)
        else: # "defect"
            face_defects = self.calculate_surface_defects_facets(use_abs=use_abs)
            curvatures = face_defects[face_ids]


        if return_num_samples:
            return origins, curvatures, len(origins)
        else:
            return origins, curvatures

    def calculate_surface_defect_vertices(self, normalize=False):
        origins = self.vertices
        defects = trimesh.curvature.vertex_defects(self.mesh)

        if normalize:
            defects /= ((self.mesh.vertex_degree)**2)
        return origins, defects

    def calculate_surface_defects_facets(self, use_abs=True):
        if self.facet_defects is not None:
            return self.facet_defects

        vertices, defects = self.calculate_surface_defect_vertices()
        # Faces take the sum of defects of vertices
        face_defects_mapped = defects[self.faces]
        if use_abs:
            face_defects = np.sum(np.abs(face_defects_mapped), axis=1)
        else:
            face_defects = np.abs(np.sum(face_defects_mapped, axis=1))
        return face_defects


    def get_vertices_of_facets(self, facet_indices):
        # Get list of faces
        facets = self.faces[facet_indices]
        # convert to 1D and remove duplicates
        vertices = set(facets.reshape(len(facets) * 3))
        return vertices

    def calculate_thicknesses_facets(self):
        trimesh.repair.fix_normals(self.mesh, multibody=True)

        num_facets = self.num_faces

        facet_offset = -self.face_normals * 0.001
        hits, ray_ids, tri_ids = self.mesh.ray.intersects_location(ray_origins=self.face_centroids + facet_offset,
                                                                   ray_directions=-self.face_normals,
                                                                   multiple_hits=False)

        hit_origins = self.face_centroids[tri_ids]
        distances = np.linalg.norm(hits - hit_origins, axis=1)
        wall_thicknesses = np.ones(num_facets) * NO_GAP_VALUE
        wall_thicknesses[tri_ids] = distances

        return wall_thicknesses

    def calculate_gap_facets(self):
        trimesh.repair.fix_normals(self.mesh, multibody=True)

        num_facets = self.num_faces

        facet_offset = self.face_normals * 0.1
        hits, ray_ids, tri_ids = self.mesh.ray.intersects_location(ray_origins=self.face_centroids + facet_offset,
                                                                   ray_directions=self.face_normals,
                                                                   multiple_hits=False)
        hit_origins = self.face_centroids[tri_ids]
        distances = np.linalg.norm(hits - hit_origins, axis=1)
        gap_sizes = np.ones(num_facets) * NO_GAP_VALUE
        gap_sizes[tri_ids] = distances
        return gap_sizes

    def get_transformed_mesh(self, scale=1.0, orientation=np.array([0, 0, 0])):
        return MeshAuxilliaryInfo(get_transformed_mesh_trs(self.mesh, scale=scale, orientation=orientation))

def create_transform_matrix(scale=np.array([1, 1, 1]),
                            translation=np.array([0, 0, 0]),
                            orientation=np.array([0, 0, 0])):
    # Order applied: translate, rotate, scale
    # orientation as [x, y, z]
    r = R.from_euler('zyx', [orientation[2], orientation[1], orientation[0]]).as_matrix()
    rot_matrix = np.zeros((4, 4))
    rot_matrix[:3, :3] = r
    rot_matrix[3, 3] = 1.0
    scale_matrix = np.diag([scale[0], scale[1], scale[2], 1.0])
    trans_matrix = np.eye(4)
    trans_matrix[:3, 3] = translation

    transform_matrix = scale_matrix @ (rot_matrix @ trans_matrix)
    return transform_matrix


def get_transformed_mesh_trs(mesh: trimesh.Trimesh, scale=np.array([1, 1, 1]), translation=np.array([0, 0, 0]),
                             orientation=np.array([0, 0, 0])):
    # Order applied: translate, rotate, scale
    # orientation as [x, y, z]
    if not isinstance(scale, np.ndarray):
        scale = np.array([scale, scale, scale])

    transform_matrix = create_transform_matrix(scale, translation, orientation)
    mesh_copy = mesh.copy()
    mesh_copy.apply_transform(transform_matrix.astype(np.float32))
    return mesh_copy

def get_largest_submesh(mesh: trimesh.Trimesh):
    if mesh.body_count > 1:
        splits = list(mesh.split(only_watertight=False))
        largest_volume = 0
        largest_submesh = None
        for submesh in splits:
            temp_volume = submesh.volume
            if temp_volume > largest_volume:
                largest_volume = temp_volume
                largest_submesh = submesh
        mesh = largest_submesh
    return mesh

def get_valid_submeshes(mesh: trimesh.Trimesh):
    valid_meshes = []
    splits = list(mesh.split(only_watertight=True))
    for submesh in splits:
        mesh_aux = MeshAuxilliaryInfo(submesh)
        if mesh_aux.is_valid:
            valid_meshes.append(submesh)
    return valid_meshes

def normalize_mesh(mesh: trimesh.Trimesh, center, normalize_scale) -> trimesh.Trimesh:
    mesh_aux = MeshAuxilliaryInfo(mesh)
    normalization_scale = 1.0
    normalization_translation = np.array([0, 0, 0])
    if center:
        centroid = np.mean(mesh_aux.vertices, axis=0)
        min_bounds = mesh_aux.bound_lower
        normalization_translation = -np.array([centroid[0], centroid[1], min_bounds[2]])
    if normalize_scale:
        scale = max(mesh_aux.bound_length)
        normalization_scale = 1.0 / scale

    mesh = get_transformed_mesh_trs(mesh, scale=normalization_scale, translation=normalization_translation)
    return mesh


##### Visualization

def show_sampled_values(mesh, points, values, normalize=True, scale=None, alpha=0.8):
    s = trimesh.Scene()
    if len(points) > 0:
        if normalize:
            values = util.normalize_minmax_01(values)
        elif scale is not None:
            values[values > scale[1]] = scale[1]
            values[values < scale[0]] = scale[0]

        cmapname = 'jet'
        cmap = plt.get_cmap(cmapname)
        colors = 255.0 * cmap(values)
        colors[:, 3] = int(alpha * 255)
        point_cloud = trimesh.points.PointCloud(vertices=points,
                                                colors=colors)
        s.add_geometry(point_cloud)
    if mesh is not None:
        s.add_geometry(mesh)
    s.show()

def show_mesh_with_normals(mesh, points, normals):
    s = trimesh.Scene()
    if len(points) > 0:
        colors = np.array([0, 0, 255, 255])
        point_cloud = trimesh.points.PointCloud(vertices=points,
                                                colors=colors)
        s.add_geometry(point_cloud)
        for i in range(len(points)):
            line = trimesh.load_path(np.array([points[i], normals[i]/3 + points[i]]))
            s.add_geometry(line)
    s.add_geometry(mesh)
    s.show()

def show_mesh_with_facet_colors(mesh, values: np.ndarray, normalize=True):
    s = trimesh.Scene()
    set_default_camera(s, mesh)
    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    empty_color = np.array([100, 100, 100, 255])

    if normalize:
        values[values != NO_GAP_VALUE] = util.normalize_minmax_01(values[values != NO_GAP_VALUE])
    mesh.visual.face_colors = cmap(values)
    mesh.visual.face_colors[values == NO_GAP_VALUE] = empty_color

    s.add_geometry(mesh)
    s.show()

def set_default_camera(scene: trimesh.Scene, mesh: trimesh.Trimesh):
    # y_min = mesh.bounds[0, 1]
    # y_max = mesh.bounds[1, 1]
    width = mesh.bounds[1, 1] - mesh.bounds[0, 1]
    height = mesh.bounds[1, 2] - mesh.bounds[0, 2]
    centroid = mesh.centroid
    camera_offset = np.array([0, -width * 2 - height, 0])
    orientation = np.array([np.pi / 2, 0, 0])
    orientation_transform = create_transform_matrix(orientation=orientation)
    translation_transform = create_transform_matrix(translation=centroid + camera_offset)
    scene.camera_transform = translation_transform @ orientation_transform

def show_mesh(mesh):
    s = trimesh.Scene()
    set_default_camera(s, mesh)
    s.add_geometry(mesh)
    s.show()

def show_meshes(meshes):
    s = trimesh.Scene()
    set_default_camera(s, meshes[0])
    for mesh in meshes:
        s.add_geometry(mesh)
    s.show()

def show_mesh_with_orientation(mesh):
    mesh_aux = MeshAuxilliaryInfo(mesh)
    colors = util.direction_to_color(mesh_aux.face_normals)
    mesh.visual.face_colors = colors
    s = trimesh.Scene()
    set_default_camera(s, mesh)
    s.add_geometry(mesh)
    s.show()

def show_mesh_with_z_normal(mesh):
    mesh_aux = MeshAuxilliaryInfo(mesh)
    colors = util.z_normal_mag_to_color(mesh_aux.face_normals)
    mesh.visual.face_colors = colors

    s = trimesh.Scene()
    set_default_camera(s, mesh)
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

