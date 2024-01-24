import os
# import tkinter.filedialog
# from tkinter import Tk     # from tkinter import Tk for Python 3.x
# from tkinter.filedialog import askopenfilename
# import torch

HOME_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
ONSHAPE_STL_PATH = HOME_PATH + "/../Onshape_STL_Dataset/"
THINGIVERSE_STL_PATH = HOME_PATH + "/../Thingiverse_STL_Dataset/"


def get_onshape_stl_path(i):
    # 34 box with holes
    # 166* Gear intricate
    # 177 hollow interior
    # 263* wedge
    # 20* intricate turbine?
    # 226* extremely intricate clock
    # 265* complex screw
    # 183* struts with ribs 1
    # 162* struts simple
    # 165* struts opening with rib holes 2
    # 167* bike frame
    return ONSHAPE_STL_PATH + "solid_" + str(i) + ".stl"

def get_thingiverse_stl_path(i, get_by_order=True):
    # 2664 chicken legs
    if get_by_order:
        contents = os.listdir(THINGIVERSE_STL_PATH)
        return THINGIVERSE_STL_PATH + contents[i]
    else:
        return THINGIVERSE_STL_PATH + str(i) + ".stl"


# def select_file(init_dir=HOME_PATH, choose_file=True):
#     Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
#     if choose_file:
#         filename = askopenfilename(initialdir=init_dir,
#                                    defaultextension="txt")  # show an "Open" dialog box and return the path to the selected file
#         return filename
#     else:
#         foldername = tkinter.filedialog.askdirectory(initialdir=init_dir)
#         return foldername

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)