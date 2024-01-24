import sys
import trimesh
import trimesh_util
import numpy as np
import paths
import random
from stopwatch import Stopwatch
import pandas as pd

def save(points, values, filename):
    names = ["x", "y", "z", "value"]
    data = np.concatenate([points, values[:, np.newaxis]], axis=1)
    df = pd.DataFrame(data=data, columns=names)
    # print("saving")
    # print(filename)
    df.to_csv(filename + ".csv", index=False, header=True)

total_stls = 10000
my_task_id = 0
num_tasks = 1

if __name__ == "__main__":
    total_stls = int(sys.argv[1])
    my_task_id = int(sys.argv[2])
    num_tasks = int(sys.argv[3])

    stopwatch = Stopwatch()

    # ## Multi STL
    stopwatch.start()
    for i in range(my_task_id, total_stls, num_tasks):
        # random_index = random.randint(0, 10000)
        # print(random_index)
        index = i
        mesh_path = paths.get_thingiverse_stl_path(index)
        mesh = trimesh.load(mesh_path)
        mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
        points, values = mesh_aux.calculate_gap_samples()

        if points is not None:
            save(points, values, paths.HOME_PATH + "generation_output/TRIMESH_" + str(index))
    stopwatch.print_time()


