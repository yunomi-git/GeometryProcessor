import trimesh_util
import trimesh
import paths
import numpy as np
from pathlib import Path
if __name__=="__main__":
    save_path = paths.DATA_PATH + "data_primitives/"
    Path(save_path).mkdir(exist_ok=True)
    primitive_name = "Cone"
    primitive_path = paths.HOME_PATH + "stls/" + primitive_name + ".stl"
    mesh = trimesh.load(primitive_path)
    min_scale = 1
    max_scale = 5
    num_data = 5000


    for i in range(num_data):
        scale = np.random.rand(3) * (max_scale - min_scale) + min_scale
        mesh_transformed = trimesh_util.get_transformed_mesh_trs(mesh, scale=scale)
        mesh_transformed.export(save_path + primitive_name + str(i) + ".stl")