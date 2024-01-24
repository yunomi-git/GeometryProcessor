import os
import sys
import paths

if __name__ == "__main__":
    folder_name = int(sys.argv[1])

    # Go into first folder
    path = paths.HOME_PATH + folder_name
    # find directory with p
    internal_dir = os.listdir(path)
    p_folder = "/"
    for folder_name in internal_dir:
        if "p" in folder_name:
            p_folder = folder_name
    path += p_folder

    max_time = 0
    rate_sum = 0

    for file in os.listdir(path):
        with open(file, 'r') as f:
            for line in f:
                if "time" in line:
                    index = line.find(":") + 1
                    time = int(line[index:])
                    if time > max_time:
                        max_time = time
                if "rate" in line:
                    index = line.find(":") + 1
                    rate = int(line[index:])
                    rate_sum += rate
    print("max_time: ", max_time)
    print("rate_sum", rate_sum)
