# if you have pyembree this should be decently fast
import matplotlib.pyplot as plt
import trimesh
import trimesh_util
import numpy as np
import paths
import random
import stopwatch
from tqdm import tqdm

NO_GAP_VALUE = -1

def get_large_areas(mesh):
    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    areas = mesh_aux.facet_areas
    median_area = np.median(areas)
    maximum_area = median_area * 3.0
    large_areas = (areas > maximum_area).nonzero()
    return large_areas
def show_large_areas(mesh):
    large_areas = get_large_areas(mesh)

    # cmapname = 'jet'
    # cmap = plt.get_cmap(cmapname)
    large_area_color = np.array([255, 100, 100, 255])
    # mesh.visual.face_colors = cmap(gap_sizes)
    mesh.visual.face_colors[large_areas] = large_area_color

    s = trimesh.Scene()
    s.add_geometry(mesh)
    s.show()

def remesh_largeareas(mesh):
    large_areas = get_large_areas(mesh)
    trimesh.remesh.subdivide(face_index=large_areas)

    s = trimesh.Scene()
    s.add_geometry(mesh)
    s.show()

if __name__ == "__main__":
    ## Single STL
    mesh_path = paths.get_onshape_stl_path(2)
    # mesh_path = 'stls/crane.stl'
    mesh = trimesh.load(mesh_path)
    # mesh = trimesh_util.TRIMESH_TEST_MESH
    show_large_areas(mesh)

    ## Multi STL
    # for i in range(20):
    #     mesh_path = paths.get_onshape_stl_path(random.randint(1, 300))
    #     mesh = trimesh.load(mesh_path)
    #     calculate_and_show_gap(mesh)
