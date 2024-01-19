import subprocess


filename = "crane"
filepath = "../stls/"

preprocessed_filename = "mesh_" + filename + ".geogram"

demo_filename = "./matfp/models/69058.stl"

# subprocess.run(["./matfp/build/MATFP_PRE", filepath + filename + ".stl", "--sub=1", "--concave=0.18", "--convex=30", "--save=1"])
subprocess.run(["./matfp/build/MATFP_PRE", demo_filename, "--sub=1", "--concave=0.18", "--convex=30", "--save=1"])

# subprocess.run(["./matfp/build/MATFP", "./matfp/input/mesh/mesh_bear.geogram", "--ds=0.1"])