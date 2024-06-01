from __future__ import print_function

from heuristic_prediction.diffusionnet_model import DiffusionNetWrapper, DiffusionNetDataset

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from heuristic_prediction.regression_tools import RegressionTools, succinct_label_save_name, seed_all
import paths
from torch.utils.data import DataLoader
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau

# system things
device = torch.device('cuda')
dtype = torch.float32

torch.cuda.empty_cache()

label_names = [
    # "overhang_violation",
    # "stairstep_violation",
    # "thickness_violation",
    # "gap_violation",
    # "volume"
    "Thickness"
    # "surface_area"
]

input_append_vertex_label_names = [
    # "nx",
    # "ny",
    # "nz"
]

input_append_global_label_names = [
    # "nx",
    # "ny",
    # "nz"
]

model_args = {
    "input_feature_type": 'xyz', # xyz, hks
    "k_eig": 64,
    "additional_dimensions": len(input_append_vertex_label_names) + len(input_append_global_label_names),
    "num_outputs": len(label_names),
    "C_width": 128,
    "N_block": 4,
    "last_activation": None,
    "outputs_at": 'vertices',
    "mlp_hidden_dims": None,
    "dropout": True,
    "with_gradient_features": True,
    "with_gradient_rotations": True,
    "diffusion_method": 'spectral',
    # "device": device
}

experiment_name = succinct_label_save_name(label_names)

args = {
    "dataset_name": "th10k_norm/train",
    "testset_name": "th10k_norm/test",
    "exp_name": experiment_name,
    "label_names": label_names,
    "input_append_vertex_label_names": input_append_vertex_label_names,
    "input_append_global_label_names": input_append_global_label_names,
    "outputs_at": "vertices",
    "seed": 1,
    "augmentations": "none",

    # Dataset Param
    "data_fraction": 0.5,
    "data_fraction_test": 0.1,
    "do_test": False,
    "workers": 24,
    "augment_random_rotate": (model_args["input_feature_type"] == 'xyz'),

    # Opt Param
    "batch_size": 1,
    "test_batch_size": 1,
    "grad_acc_steps": 16,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-5,

    "scheduler": "plateau",
    # "restarts": 3,
    # "min_lr": 5e-4,
    "patience": 5,
    "factor": 0.1
}

args.update(model_args)

if __name__=="__main__":
    seed_all(args["seed"])
    data_root_dir = paths.DATA_PATH + args["dataset_name"] + "/"
    test_root_dir = paths.DATA_PATH + args["testset_name"] + "/"
    op_cache_dir = data_root_dir + "op_cache"

    train_loader = DataLoader(DiffusionNetDataset(data_root_dir, model_args["k_eig"],
                                                  op_cache_dir=op_cache_dir,
                                                  data_fraction=args["data_fraction"], label_names=label_names,
                                                  augment_random_rotate=args["augment_random_rotate"],
                                                  extra_vertex_label_names=args["input_append_vertex_label_names"],
                                                  extra_global_label_names=args["input_append_global_label_names"],
                                                  outputs_at=args["outputs_at"],
                                                  is_training=True,
                                                  cache_operators=False),
                                  num_workers=24,
                                  batch_size=args['batch_size'], shuffle=True, drop_last=True)
    test_loader = None
    if args["do_test"]:
        test_loader = DataLoader(DiffusionNetDataset(test_root_dir, model_args["k_eig"],
                                                     op_cache_dir=op_cache_dir,
                                                     data_fraction=args["data_fraction"], label_names=label_names,
                                                     augment_random_rotate=args["augment_random_rotate"],
                                                     extra_vertex_label_names=args["input_append_vertex_label_names"],
                                                     extra_global_label_names=args["input_append_global_label_names"],
                                                     outputs_at=args["outputs_at"],
                                                     is_training=False),
                                 num_workers=24,
                                 batch_size=args['test_batch_size'], shuffle=True, drop_last=False)

    model = DiffusionNetWrapper(args, op_cache_dir, device)
    model = model.to(device)

    # === Optimize
    opt = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args["weight_decay"])
    scheduler = None
    if args["scheduler"] == "cosine_annealing":
        if args["restarts"] == 0:
            scheduler = CosineAnnealingLR(opt, args["epochs"], eta_min=args["min_lr"])
        else:
            scheduler = CosineAnnealingWarmRestarts(opt, args["epochs"], T_mult=args["restarts"], eta_min=args["min_lr"])
    if args["scheduler"] == "plateau":
            scheduler = ReduceLROnPlateau(opt, patience=args["patience"], factor=args["factor"])


    regression_manager = RegressionTools(
        args=args,
        label_names=label_names,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        opt=opt,
        scheduler=scheduler,
        clip_parameters=True,
        include_faces=True
    )

    regression_manager.train(args, do_test=args["do_test"], plot_every_n_epoch=1)






