import skeletor as sk
import trimesh as tm

# mesh = sk.example_mesh()
# To load and use your own mesh instead of the example mesh:
# mesh = tm.Trimesh(vertices, faces)  # or...

mesh = tm.load_mesh('../stls/Antenna_DJI.stl')

fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
skel = sk.skeletonize.by_wavefront(fixed, waves=500, step_size=20)
skel.show(mesh=True)