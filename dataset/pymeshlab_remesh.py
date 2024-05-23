import pymeshlab
import paths
from tqdm import tqdm
import util
import trimesh
import trimesh_util
import numpy as np
import os
from pathlib import Path
import FolderManager

time = util.Stopwatch()

# mesh = trimesh.load(paths.HOME_PATH + "stls/kunai.stl")
# mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
# facet_centroids = mesh_aux.facet_centroids
# num_facets = len(facet_centroids)
# face_ids = np.arange(num_facets)
# _, facet_curvatures = mesh_aux.calculate_curvature_at_points(origins=facet_centroids, face_ids=face_ids, curvature_method="defect", use_abs=True)
# # facet_curvatures[facet_curvatures>0.3]=0.3
# # trimesh_util.show_mesh_with_facet_colors(mesh, facet_curvatures)
# high_curvature = util.get_indices_of_conditional(facet_curvatures > 1)
# low_curvature = util.get_indices_of_conditional(facet_curvatures < 0.5)
# labmesh = pymeshlab.Mesh(vertex_matrix=mesh_aux.vertices, face_matrix=mesh_aux.facets)

# def create_face_selection(indices):
#     string = ""
#     for index in indices:
#         if len(string) > 0:
#             string += "||"
#
#         string += "(vi0==%d)||(vi1==%d)||(vi2==%d)"% (index, index, index)
#     return string
#
# selection_string = create_face_selection(low_curvature)


def curvature_based_remesh_old(file_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_path)
    time.start()

    # First obtain minimum resolution
    ms.meshing_surface_subdivision_midpoint(threshold = pymeshlab.PercentageValue(2))

    # At good resolution, calculate curvatures
    ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype = 3)

    # Then collapse large faces
    ms.compute_selection_by_color_per_face(color = pymeshlab.Color(255, 0, 0, 255),
                                           percentrh = 0.1, percentgs = 0.1, percentbv = 0.2)
    ms.meshing_isotropic_explicit_remeshing(iterations = 3,
                                            selectedonly = True,
                                            targetlen = pymeshlab.PercentageValue(2),
                                            splitflag=True,
                                            collapseflag = True)

    # Also refine medium faces
    ms.compute_selection_by_color_per_face(color = pymeshlab.Color(38, 255, 0, 255),
                                           percentrh = 0.25)
    ms.meshing_isotropic_explicit_remeshing(iterations = 3,
                                            selectedonly = True,
                                            targetlen = pymeshlab.PercentageValue(1),
                                            splitflag=True, collapseflag = True)

    time.print_time()
    ms.show_polyscope()

def curvature_based_remesh(file_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_path)
    time.start()

    # First obtain minimum resolution
    ms.meshing_surface_subdivision_midpoint(iterations=3,
                                            threshold=pymeshlab.PercentageValue(0.5))

    # At good resolution, calculate curvatures
    ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype = 3)

    # Then collapse large faces
    ms.compute_selection_by_color_per_face(color = pymeshlab.Color(255, 0, 0, 255),
                                           percentrh = 0.1, percentgs = 0.1, percentbv = 0.2)
    ms.meshing_isotropic_explicit_remeshing(iterations = 3,
                                            selectedonly = True,
                                            targetlen = pymeshlab.PercentageValue(1),
                                            splitflag=True,
                                            collapseflag = True)

    # Also refine medium faces
    ms.apply_selection_inverse(invfaces=True)
    ms.meshing_isotropic_explicit_remeshing(iterations = 3,
                                            selectedonly = True,
                                            targetlen = pymeshlab.PercentageValue(0.5),
                                            splitflag=True, collapseflag = True)

    time.print_time()
    ms.show_polyscope()

def default_remesh(file_path, out_path, show=False):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_path)
    # time.start()
    ms.meshing_surface_subdivision_midpoint(iterations=3)
    ms.meshing_isotropic_explicit_remeshing(iterations=5)
    # ms.meshing_isotropic_explicit_remeshing(iterations=6,
    #                                         targetlen=pymeshlab.PercentageValue(1),
    #                                         splitflag=True,
    #                                         collapseflag=True,
    #                                         reprojectflag=True)
    # time.print_time()0
    if out_path is not None:
        ms.save_current_mesh(out_path, save_face_color=False)
    if show:
        ms.show_polyscope()



if __name__=="__main__":
    original_path = paths.HOME_PATH + "../Datasets/" + "Thingi10k_Normalized/"
    new_path = paths.HOME_PATH + "../Datasets/" + "Thingi10k_Remesh_Normalized/"
    Path(new_path).mkdir(exist_ok=True)

    file_names = os.listdir(original_path)
    file_names.sort()
    original_extension = file_names[0][file_names[0].find("."):]
    file_names = [file_name[:file_name.find(".")] for file_name in file_names]
    start_from = 0
    for file_name in tqdm(file_names[start_from:]):
        try:
            mesh = trimesh.load(original_path + file_name + original_extension)
            mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
        except:
            print("error loading")
            continue

        if not mesh_aux.is_valid:
            continue

        try:

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(original_path + file_name + original_extension)

            if not mesh.is_watertight:
                continue
            else:
                ms.meshing_surface_subdivision_midpoint(iterations=3)
                ms.meshing_isotropic_explicit_remeshing(iterations=5)

            ms.save_current_mesh(new_path + file_name + ".stl", save_face_color=False)
        except:
            print("error remeshing")
            continue




        # default_remesh(file_path=original_path + file_name + original_extension,
        #                out_path=new_path + file_name + ".stl")