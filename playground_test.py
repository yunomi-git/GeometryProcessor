import trimesh
import trimesh_util
import paths
from datetime import datetime
import sklearn.metrics as metrics
import numpy as np


# mesh = trimesh.load(paths.HOME_PATH + "stls/shell_3.stl")
# splits = list(mesh.split(only_watertight=False))
# for body in splits:
#     trimesh_util.show_mesh(body)

# current = datetime.now()
# encode = "%d%d_%d%d" % (current.month, current.day, current.hour, current.minute)
# print(encode)

a = np.array([[1,2,3,4,5],[1,2,3,4,5]]).T
b = np.array([[1,2,3,4,6],[1,2,3,4,5]]).T
r2 = metrics.r2_score(a, b, multioutput='raw_values')
print(r2)
