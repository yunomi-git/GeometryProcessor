from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import numpy as np
import math3d
from matplotlib import cm
import vtkplotlib as vpl

if __name__ == "__main__":
    mesh_to_analyze = mesh.Mesh.from_file('stls/kunai.stl')



    centroids = mesh_to_analyze.centroids
    normals = mesh_to_analyze.normals
    areas = mesh_to_analyze.areas
    characteristic_radii = np.sqrt(areas / np.pi)
    num_facets = len(mesh_to_analyze.data)

    is_thin_vector = np.empty(num_facets)
    wall_thicknesses = np.empty(num_facets)


    for i in range(num_facets):
        # First find the opposing centroid. 1. check for intersecting facets 2.  check for closest of intersecting
        print(i)

        # Over all other centroids, find the ones that intersect
        distances = math3d.distance_point_from_axis(centroids[i], axis=normals[i], point=centroids)
        distances = distances[distances > 0]

        # is_intersecting = np.logical_and(distances.flatten() < characteristic_radii.flatten(), distances.flatten() > 0)
        # intersecting_centroids = centroids[is_intersecting]
        #
        # # Over intersecting centroids, find one with minimum distance
        # axis_distances = math3d.point_onto_axis_distance_along_axis(centroids[i], axis=normals[i], point=intersecting_centroids)
        # closest_centroid = axis_distances.index(min(axis_distances))
        # wall_thicknesses[i] = min(axis_distances)

        wall_thicknesses[i] = min(distances)



    # poly_other = mplot3d.art3d.Poly3DCollection(other_vectors)
    # poly_other.set_facecolor((0.5, 0.5, 0, 0.30))
    # axes.add_collection3d(poly_other)
    # scale = mesh_to_analyze.points.flatten()
    # axes.auto_scale_xyz(scale, scale, scale)
    # # axes.autoscale(True)

    vpl.mesh_plot(mesh_to_analyze, tri_scalars=wall_thicknesses)
    vpl.show()

