import trimesh
import trimesh_util
import paths

mesh = trimesh.load(paths.HOME_PATH + "stls/shell_3.stl")
splits = list(mesh.split(only_watertight=False))
for body in splits:
    trimesh_util.show_mesh(body)