import pymeshlab
import paths
import util
import trimesh
import trimesh_util
import numpy as np

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




ms = pymeshlab.MeshSet()
ms.load_new_mesh(paths.HOME_PATH + "stls/crane.stl")
# ms.add_mesh(labmesh)
time.start()

# First obtain minimum resolution
ms.meshing_surface_subdivision_midpoint(threshold = pymeshlab.PercentageValue(2))
# ms.meshing_isotropic_explicit_remeshing(smoothflag=True,
#                                         # adaptive=True,
#                                         iterations=3,
#                                         splitflag=True,
#                                         featuredeg=30,
#                                         checksurfdist =True,
#                                         collapseflag=False,
#                                         # swapflag=False,
#                                         # selectedonly =True,
#                                         targetlen=pymeshlab.PercentageValue(10),
#                                         # maxsurfdist=pymeshlab.PercentageValue(0.5)
#                                         )

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

# ms.compute_selection_by_color_per_face(color = pymeshlab.Color(0, 255, 0, 255))
# ms.meshing_isotropic_explicit_remeshing(iterations = 3,
#                                         selectedonly = True,
#                                         targetlen = pymeshlab.PercentageValue(0.5),
#                                         splitflag=True, collapseflag = True)

time.print_time()
ms.show_polyscope()