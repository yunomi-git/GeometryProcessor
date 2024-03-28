from __future__ import print_function
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from heuristic_prediction.regression_tools import RegressionTools, succinct_label_save_name
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
    # "volume"
    "thickness"
]

model_args = {
    "num_points": 2048,
    "conv_channel_sizes": [128, 128, 256, 512],  # Default: [64, 64, 128, 256] #Mo: [512, 512, 1024]
    "emb_dims": 256,
    "linear_sizes": [1024, 512, 256, 128, 64, 32, 16],  # [512, 256] #Mo: [1024, 512, 256, 128, 64, 32, 16]
    "num_outputs": len(label_names),
    "k": 20,
    "dropout": 0.2,
    "outputs_at": "vertices"
}

experiment_name = succinct_label_save_name(label_names)

args = {
    "exp_name": experiment_name,
    "label_names": label_names,

    # Dataset Param
    "data_fraction": 0.05,
    "data_fraction_test": 0.15,
    "workers": 24,
    "grad_acc_steps": 2,
    "normalize_inputs": False,
    "sampling_method": "mixed",

    # Opt Param
    "batch_size": 8,
    "test_batch_size": 8,
    "epochs": 30,
    "lr": 1e-2,
    "min_lr": 5e-5,
    "weight_decay": 1e-4,
    "seed": 1,
}

args.update(model_args)

if __name__ == "__main__":
    ### Data ###
    data_root_dir = paths.DATA_PATH + "data_th5k_norm/"
    train_loader = DataLoader(PointCloudDataset(data_root_dir, args['num_points'], label_names=label_names,
                                                partition='train',
                                                # filter_criteria=filter_criteria, use_augmentations=True,
                                                data_fraction=args["data_fraction"], use_numpy=True,
                                                normalize=args["normalize_inputs"],
                                                sampling_method=args["sampling_method"],
                                                outputs_at=args["outputs_at"]),
                              num_workers=24,
                              batch_size=args['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(PointCloudDataset(data_root_dir, args['num_points'], label_names=label_names,
                                               partition='test',
                                               # filter_criteria=filter_criteria, use_augmentations=False,
                                               data_fraction=args["data_fraction_test"], use_numpy=True,
                                               normalize=args["normalize_inputs"],
                                               sampling_method=args["sampling_method"],
                                               outputs_at=args["outputs_at"]),
                             num_workers=24,
                             batch_size=args['test_batch_size'], shuffle=True, drop_last=False)

    ### Model ###
    if args["outputs_at"] == "global":
        model = DGCNN_param(args)
    else: # "vertices"
        model = DGCNN_segment(args)
    # opt = torch.optim.SGD(model.parameters(), lr=args["lr"], momentum=0.9, weight_decay=args["weight_decay"])
    opt = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args["weight_decay"])
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

    regression_manager.train(args, do_test=False, plot_every_n_epoch=1)
