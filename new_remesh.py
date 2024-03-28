import pymeshlab
import paths
import util
import trimesh
import trimesh_util
import numpy as np

time = util.Stopwatch()

mesh = trimesh.load(paths.HOME_PATH + "stls/kunai.stl")
mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
facet_centroids = mesh_aux.facet_centroids
num_facets = len(facet_centroids)
face_ids = np.arange(num_facets)
_, facet_curvatures = mesh_aux.calculate_curvature_at_points(origins=facet_centroids, face_ids=face_ids, curvature_method="defect", use_abs=True)
# facet_curvatures[facet_curvatures>0.3]=0.3
# trimesh_util.show_mesh_with_facet_colors(mesh, facet_curvatures)
high_curvature = util.get_indices_of_conditional(facet_curvatures > 1)
low_curvature = util.get_indices_of_conditional(facet_curvatures < 0.5)
labmesh = pymeshlab.Mesh(vertex_matrix=mesh_aux.vertices, face_matrix=mesh_aux.facets)

def create_face_selection(indices):
    string = ""
    for index in indices:
        if len(string) > 0:
            string += "||"

        string += "(vi0==%d)||(vi1==%d)||(vi2==%d)"% (index, index, index)
    return string

selection_string = create_face_selection(low_curvature)




ms = pymeshlab.MeshSet()
ms.load_new_mesh(paths.HOME_PATH + "stls/kunai.stl")
# ms.add_mesh(labmesh)
time.start()
# ms.compute_selection_by_condition_per_face(condselect =selection_string)

# ms.compute_selection_by_condition_per_face(condselect ="(q0 < 0.01)||(q1 < 0.01)||(q2 < 0.01)")
ms.compute_curvature_and_color_rimls_per_vertex()
a = ms.mesh(0).face_color_matrix()
# ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype="Mean Curvature")
ms.compute_selection_by_color_per_face(color=pymeshlab.Color(r=255, g=100, b=100),
                                        percentrh=0.5,
                                       percentgs=1.0,
                                       percentbv=1.0)

ms.meshing_isotropic_explicit_remeshing(smoothflag=True,
                                        # adaptive=True,
                                        iterations=3,
                                        splitflag=True,
                                        featuredeg=30,
                                        checksurfdist =True,
                                        collapseflag=False,
                                        # swapflag=False,
                                        # selectedonly =True,
                                        targetlen=pymeshlab.PercentageValue(10),
                                        # maxsurfdist=pymeshlab.PercentageValue(0.5)
                                        )
# ms.meshing_isotropic_explicit_remeshing(smoothflag=True,
#                                         # adaptive=True,
#                                         iterations=3,
#                                         splitflag=False,
#                                         featuredeg=30,
#                                         checksurfdist =True,
#                                         collapseflag=True,
#                                         # swapflag=False,
#                                         # selectedonly =True,
#                                         targetlen=pymeshlab.PercentageValue(0.1),
#                                         # maxsurfdist=pymeshlab.PercentageValue(0.5)
#                                         )
# ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(5))
# ms.meshing_surface_subdivision_midpoint(iterations=3,
#                                          threshold=pymeshlab.PercentageValue(1))
time.print_time()
ms.show_polyscope()