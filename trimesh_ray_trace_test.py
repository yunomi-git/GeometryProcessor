# if you have pyembree this should be decently fast
import matplotlib.pyplot as plt
import trimesh
import trimesh_util
import numpy as np
import paths

if __name__ == "__main__":
    # mesh_path = paths.get_onshape_stl_path(2)
    mesh_path = 'stls/crane.stl'
    mesh = trimesh.load(mesh_path)
    # mesh = trimesh_util.TRIMESH_TEST_MESH
    # mesh.fix_normals(multibody=False)
    trimesh.repair.fix_normals(mesh, multibody=True)
    # trimesh.repair.fix_inversion(mesh, multibody=True)
    # trimesh.repair.fix_winding(mesh)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    num_facets = mesh_aux.num_facets

    is_thin_vector = np.empty(num_facets)

    hits = mesh.ray.intersects_location(ray_origins=mesh_aux.facet_centroids,
                                        ray_directions=-mesh_aux.facet_normals,
                                        multiple_hits=False)[0]
    distances = np.linalg.norm(hits - mesh_aux.facet_centroids, axis=1)
    wall_thicknesses = distances

    # Now normalize
    wall_thicknesses -= np.amin(wall_thicknesses)
    max_thickness = np.amax(wall_thicknesses)
    wall_thicknesses /= max_thickness

    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    colors = cmap(wall_thicknesses)
    mesh.visual.face_colors = cmap(wall_thicknesses)

    s = trimesh.Scene()
    s.add_geometry(mesh)
    # Add the voxelized mesh to the scene. If want to also show the intial mesh uncomment the second line and change the alpha channel of in the loop to something <100
    # s.add_geometry(voxels)
    s.show()