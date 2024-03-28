from __future__ import print_function


from heuristic_prediction.diffusionnet_model import DiffusionNetData, DiffusionNetWrapper, DiffusionNetDataset

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from heuristic_prediction.regression_tools import RegressionTools, succinct_label_save_name
import paths
from torch.utils.data import DataLoader
import math

split_size = 10

# system things
device = torch.device('cuda')
dtype = torch.float32

torch.cuda.empty_cache()

label_names = [
    # "overhang_violation",
    # "stairstep_violation",
    "thickness_violation",
    # "gap_violation",
    # "volume"
]

model_args = {
    "input_feature_type": 'hks', # xyz, hks
    "k_eig": 64,
    "num_outputs": len(label_names),
    "C_width": 128,
    "N_block": 4,
    "last_activation": None,
    "outputs_at": 'vertices',
    "mlp_hidden_dims": None,
    "dropout": True,
    "with_gradient_features": True,
    "with_gradient_rotations": True,
    "diffusion_method": 'spectral'
}

experiment_name = succinct_label_save_name(label_names)

args = {
    "exp_name": experiment_name,
    "label_names": label_names,

    # Dataset Param
    "data_fraction": 0.3,
    "data_fraction_test": 0.15,
    "workers": 24,
    "grad_acc_steps": 2,
    "normalize_inputs": False,
    "sampling_method": "mixed",
    "augment_random_rotate": (model_args["input_feature_type"] == 'xyz'),

    # Opt Param
    "batch_size": 8,
    "test_batch_size": 8,
    "epochs": 100,
    "lr": 1e-3,
    # "min_lr": 5e-5,
    "num_lr_reductions": 8,
    "lr_reduction_gamma": 0.5,
    "weight_decay": 1e-4,
    "seed": 1,
}

args.update(model_args)

# training settings
num_decays = 8
# decay_every = int(n_epoch / num_decays) #50
decay_rate = 0.5
# label_smoothing_fac = 0.2

def filter_criteria(mesh, instance_data) -> bool:
    if instance_data["vertices"] < 1e3:
        return False
    if instance_data["vertices"] > 1e5:
        return False

    return True

if __name__=="__main__":
    dataset_path = paths.DATA_PATH + "data_th5k_norm/"
    op_cache_dir = dataset_path + "op_cache"

    train_loader = DataLoader(DiffusionNetDataset(dataset_path, split_size, model_args["k_eig"],
                                                 filter_criteria=filter_criteria, op_cache_dir=op_cache_dir,
                                                 data_fraction=args["data_fraction"], label_names=label_names,
                                                 augment_random_rotate=args["augment_random_rotate"],
                                                 is_training=True),
                              num_workers=24,
                              batch_size=args['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(DiffusionNetDataset(dataset_path, split_size, model_args["k_eig"],
                                                 filter_criteria=filter_criteria, op_cache_dir=op_cache_dir,
                                                 data_fraction=args["data_fraction_test"], label_names=label_names,
                                                 augment_random_rotate=args["augment_random_rotate"],
                                                 is_training=False),
                             num_workers=24,
                             batch_size=args['test_batch_size'], shuffle=True, drop_last=False)

    model = DiffusionNetWrapper(input_feature_type=model_args["input_feature_type"],
                                num_outputs=model_args["num_outputs"],
                                C_width=model_args["C_width"], N_block=model_args["N_block"],
                                last_activation=model_args["last_activation"], outputs_at=model_args["outputs_at"],
                                mlp_hidden_dims=model_args["mlp_hidden_dims"], dropout=model_args["dropout"],
                                with_gradient_features=model_args["with_gradient_features"],
                                with_gradient_rotations=model_args["with_gradient_rotations"],
                                diffusion_method=model_args["diffusion_method"])
    model = model.to(device)

    # === Optimize
    opt = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args["weight_decay"])
    # scheduler = CosineAnnealingLR(opt, args["epochs"], eta_min=args["min_lr"])
    scheduler = StepLR(opt, step_size=int(args["epochs"] / args["num_lr_reductions"]), gamma=args["lr_reduction_gamma"])

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






