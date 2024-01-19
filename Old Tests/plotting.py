import vtkplotlib as vpl
from stl.mesh import Mesh

if __name__=="__main__":
    mesh_to_analyze = Mesh.from_file('../stls/low-res.stl')
    centroids = mesh_to_analyze.centroids
    z_heights = centroids[:, 2]
    vpl.mesh_plot(mesh_to_analyze, tri_scalars=z_heights)
    vpl.show()