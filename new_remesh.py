import pymeshlab
import paths
import util
import trimesh
import trimesh_util
import numpy as np

time = util.Stopwatch()

mesh = trimesh.load(paths.HOME_PATH + "stls/crane.stl")
mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
facet_centroids = mesh_aux.facet_centroids
num_facets = len(facet_centroids)
face_ids = np.arange(num_facets)
_, facet_curvatures = mesh_aux.calculate_curvature_at_points(origins=facet_centroids, face_ids=face_ids, curvature_method="defect", use_abs=True)
# facet_curvatures[facet_curvatures>0.3]=0.3
trimesh_util.show_mesh_with_facet_colors(mesh, facet_curvatures)
high_curvature = util.get_indices_of_conditional(facet_curvatures > 1)
low_curvature = util.get_indices_of_conditional(facet_curvatures < 0.01)
labmesh = pymeshlab.Mesh(vertex_matrix=mesh_aux.vertices, face_matrix=mesh_aux.facets)



ms = pymeshlab.MeshSet()
# ms.load_new_mesh(paths.HOME_PATH + "stls/kunai.stl")
ms.add_mesh(labmesh)
time.start()
ms.compute_selection_by_condition_per_face(condselect=facet_curvatures > 1)
ms.meshing_isotropic_explicit_remeshing(smoothflag=True,
                                        adaptive=True,
                                        iterations=4,
                                        splitflag=True,
                                        featuredeg=30,
                                        targetlen=pymeshlab.PercentageValue(0.5),
                                        # maxsurfdist=pymeshlab.PercentageValue(0.5)
                                        )
# ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(5))
# ms.meshing_surface_subdivision_midpoint(iterations=3,
#                                          threshold=pymeshlab.PercentageValue(1))
time.print_time()
ms.show_polyscope()