from __future__ import print_function
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from heuristic_prediction.regression_tools import RegressionTools, succinct_label_save_name, seed_all
import paths
from heuristic_prediction.dgcnn_model import DGCNN_param, DGCNN_segment
from torch.utils.data import DataLoader
from heuristic_prediction.pointcloud_dataloader import PointCloudDataset

torch.cuda.empty_cache()

label_names = [
    # "overhang_violation",
    # "stairstep_violation",
    # "thickness_violation",
    # "gap_violation",
    # "volume",
    "surface_area"
    # "thickness",
    # "nx",
    # "ny",
    # "nz"
]

model_args = {
    "num_points": 2048,
    "conv_channel_sizes": [128, 128, 256, 512],  # Default: [64, 64, 128, 256] #Mo: [512, 512, 1024]
    "emb_dims": 512,
    "linear_sizes": [1024, 512, 256, 128, 64, 32, 16],  # [512, 256] #Mo: [1024, 512, 256, 128, 64, 32, 16]
    "num_outputs": len(label_names),
    "k": 20,
    "dropout": 0.2,
    "outputs_at": "global"
}

experiment_name = succinct_label_save_name(label_names)

args = {
    "exp_name": experiment_name,
    "label_names": label_names,

    # Dataset Param
    "data_fraction": 0.3,
    "data_fraction_test": 0.1 / 4,
    "workers": 24,
    "grad_acc_steps": 4,
    "normalize_inputs": False,
    "sampling_method": "mixed",
    "imbalanced_weighting_bins": 5, #1 means no weighting
    "do_test": False,
    "normalize_outputs": False,

    # Opt Param
    "batch_size": 8,
    "test_batch_size": 8,
    "epochs": 50,
    "lr": 1e-2,
    "min_lr": 5e-4,
    "weight_decay": 1e-5,
    "seed": 1,
    "restarts": 3,
}

args.update(model_args)

if __name__ == "__main__":
    ### Data ###
    seed_all(args["seed"])
    data_root_dir = paths.DATA_PATH + "data_th5k_norm/"
    train_loader = DataLoader(PointCloudDataset(data_root_dir, args['num_points'], label_names=label_names,
                                                partition='train',
                                                # filter_criteria=filter_criteria, use_augmentations=True,
                                                data_fraction=args["data_fraction"], use_numpy=True,
                                                normalize=args["normalize_inputs"],
                                                normalize_outputs=args["normalize_outputs"],
                                                sampling_method=args["sampling_method"],
                                                outputs_at=args["outputs_at"],
                                                imbalance_weight_num_bins=args["imbalanced_weighting_bins"]),
                              num_workers=24,
                              batch_size=args['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(PointCloudDataset(data_root_dir, args['num_points'], label_names=label_names,
                                               partition='test',
                                               # filter_criteria=filter_criteria, use_augmentations=False,
                                               data_fraction=args["data_fraction_test"], use_numpy=True,
                                               normalize=args["normalize_inputs"],
                                               normalize_outputs=False,
                                               sampling_method=args["sampling_method"],
                                               outputs_at=args["outputs_at"],
                                               imbalance_weight_num_bins=args["imbalanced_weighting_bins"]),
                             num_workers=24,
                             batch_size=args['test_batch_size'], shuffle=True, drop_last=False)

    ### Model ###
    if args["outputs_at"] == "global":
        model = DGCNN_param(args)
    else: # "vertices"
        model = DGCNN_segment(args)
    # opt = torch.optim.SGD(model.parameters(), lr=args["lr"], momentum=0.9, weight_decay=args["weight_decay"])
    opt = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args["weight_decay"])
    if args["restarts"] == 0:
        scheduler = CosineAnnealingLR(opt, args["epochs"], eta_min=args["min_lr"])
    else:
        scheduler = CosineAnnealingWarmRestarts(opt, args["epochs"], T_mult=args["restarts"], eta_min=args["min_lr"])




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

    regression_manager.train(args, do_test=False, plot_every_n_epoch=1, outputs_at=args["outputs_at"])
