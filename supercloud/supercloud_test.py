import sys
#import trimesh
#import trimesh_util
import numpy as np
# import paths
# import random
# from util import Stopwatch
# import pandas as pd
from pathlib import Path

# def save(points, values, filename):
#     names = ["x", "y", "z", "value"]
#     data = np.concatenate([points, values[:, np.newaxis]], axis=1)
#     df = pd.DataFrame(data=data, columns=names)
#     # print("saving")
#     # print(filename)
#     df.to_csv(filename + ".csv", index=False, header=True)

# total_stls = 10000
# my_task_id = 0
# num_tasks = 1

if __name__ == "__main__":

    total_stls = int(sys.argv[1])
    my_task_id = int(sys.argv[2])
    num_tasks = int(sys.argv[3])
    print("Starting", my_task_id)
    # stopwatch = Stopwatch()

    # ## Multi STL
    count = 0
    # stopwatch.start()
    save_path = "supercloud/test/"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for i in range(my_task_id, total_stls, num_tasks):
        with open(save_path + str(i) + "_" + str(my_task_id)) as f:
            f.write("hello")
        # count += 1
        # # random_index = random.randint(0, 10000)
        # # print(random_index)
        # index = i
        # mesh_path = paths.get_thingiverse_stl_path(index)
        # mesh = trimesh.load(mesh_path)
        # mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
        # points, values = mesh_aux.calculate_gap_samples()
        #
        # if points is not None:
        #     save(points, values, paths.HOME_PATH + "generation_output/TRIMESH_" + str(index))
    # time = stopwatch.get_time()
    # print("count: ", count)
    # print("time: ", time)
    # print("rate: ", count / time)


