import trimesh
import trimesh_util
import paths

def play_mirror_surface():
    mesh_path = paths.RAW_DATASETS_PATH + 'DrivAerNet/N_S_WW_WM_3/Exp_001/N_S_WW_WM_3.stl'
    mesh = trimesh.load(mesh_path)

    mirror_car = trimesh_util.mirror_surface(mesh, plane="y")
    print("watertight", mirror_car.is_watertight)

    mirror_aux = trimesh_util.MeshAuxilliaryInfo(mirror_car)

    trimesh_util.show_mesh(mirror_aux.mesh)

if __name__=="__main__":
    play_mirror_surface()