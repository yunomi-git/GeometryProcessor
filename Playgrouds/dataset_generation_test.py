import trimesh
import trimesh_util
import paths
import numpy as np
from heuristic_prediction import printability_metrics
from tqdm import tqdm
import json
import os
from pathlib import Path
import pymeshlab


if __name__=="__main__":
    # file_path = paths.HOME_PATH + "../Datasets/Dataset_Thingiverse_10k/"
    # contents = os.listdir(file_path)
    # contents.sort()
    # source_mesh_filenames = [file_path + file_name for file_name in contents]
    #
    # # load mesh
    # # remesh
    # # load vertices
    # # calculate values from vertices
    # # save vertices and faces

    def default_remesh(file_path, show=False):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file_path)

        # First obtain minimum resolution
        ms.meshing_surface_subdivision_midpoint(iterations=3,
                                                threshold=pymeshlab.PercentageValue(0.5))

        # At good resolution, calculate curvatures
        ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=3)

        # Then collapse large faces
        ms.compute_selection_by_color_per_face(color=pymeshlab.Color(255, 0, 0, 255),
                                               percentrh=0.1, percentgs=0.1, percentbv=0.2)
        ms.apply_selection_inverse(invfaces=True)
        ms.meshing_isotropic_explicit_remeshing(iterations=6,
                                                selectedonly=True,
                                                targetlen=pymeshlab.PercentageValue(0.5),
                                                splitflag=True, collapseflag=True)
        ms.apply_selection_inverse(invfaces=True)
        ms.meshing_isotropic_explicit_remeshing(iterations=6,
                                                selectedonly=True,
                                                targetlen=pymeshlab.PercentageValue(1),
                                                splitflag=True,
                                                collapseflag=True)

        # Also refine medium faces

        if show:
            ms.show_polyscope()

    default_remesh(paths.HOME_PATH + "stls/100139.stl", show=True)

