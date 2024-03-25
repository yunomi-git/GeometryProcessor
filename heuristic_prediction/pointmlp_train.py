import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from heuristic_prediction import pointmlp_model

import paths
import math
from heuristic_prediction.mesh_dataloader import PointCloudDataset
from heuristic_prediction.regression_tools import RegressionTools, succinct_label_save_name



label_names = [
    # "overhang_violation",
    # "stairstep_violation",
    # "thickness_violation",
    # "gap_violation"
    # "volume",
    # "bound_length",
    "centroid_x"
    # "surface_area"
]

experiment_name = succinct_label_save_name(label_names)

model_args = {
    "num_points": 2048,
    "emb_dims": 256, # 64
    "num_outputs": len(label_names),
    "groups": 1, #1
    "res_expansion": 1.0, # 1.0
    "activation": "relu", #"relu"

    "bias": False,
    "use_xyz": False,
    "normalize": "anchor",
    "dim_expansion": [2, 2, 2, 2],
    "pre_blocks": [2, 2, 2, 2],
    "pos_blocks":[2, 2, 2, 2],
    "k_neighbors":[24, 24, 24, 24],
    "reducers":[2, 2, 2, 2]
}

args = {
    # Dataset Param
    "data_fraction": 1.0,
    "data_fraction_test": 0.006,
    "workers": 24,
    "grad_acc_steps": 1,
    "normalize_inputs": True,

    # Opt Param
    "batch_size": 32,
    "test_batch_size": 8,
    "epochs": 100,
    "lr": 1e-2,
    "min_lr": 5e-3,
    "weight_decay": 2e-4,
    "seed": 1,
    # "gradient_clip": 1,

    "exp_name": experiment_name,
    "label_names": label_names,
}

args.update(model_args)

def filter_criteria(mesh, instance_data) -> bool:
    # if instance_data["vertices"] < 1e3:
    #     return False
    # if instance_data["vertices"] > 1e5:
    #     return False
    if math.isnan(instance_data["thickness_violation"]):
        return False
    # if instance_data["scale"] > 1000:
    #     return False

    return True

def main():
    ### Data ###
    data_root_dir = paths.HOME_PATH + "data_th5k_aug/"# "data_augmentations/" #
    train_loader = DataLoader(PointCloudDataset(data_root_dir, args['num_points'], label_names=label_names,
                                                partition='train',
                                                filter_criteria=filter_criteria, use_augmentations=True,
                                                data_fraction=args["data_fraction"], use_numpy=True,
                                                normalize=args["normalize_inputs"]),
                              num_workers=24,
                              batch_size=args['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(PointCloudDataset(data_root_dir, args['num_points'], label_names=label_names,
                                               partition='test',
                                               filter_criteria=filter_criteria, use_augmentations=False,
                                               data_fraction=args["data_fraction_test"], use_numpy=True,
                                               normalize=args["normalize_inputs"]),
                             num_workers=24,
                             batch_size=args['test_batch_size'], shuffle=True, drop_last=False)

    ### Model ###
    model = pointmlp_model.defaultPointMLPModel(args)

    # opt = torch.optim.SGD(model.parameters(), lr=args["lr"], momentum=0.9, weight_decay=args["weight_decay"])
    opt = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args["weight_decay"])
    scheduler = CosineAnnealingLR(opt, args["epochs"], eta_min=args["min_lr"])


    regression_manager = RegressionTools(
        args=args,
        label_names=label_names,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        opt=opt,
        scheduler=scheduler,
        clip_parameters=True
    )

    regression_manager.train(args)


if __name__ == '__main__':
    main()
