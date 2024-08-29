from __future__ import print_function

from shape_regression.diffusionnet_model import DiffusionNetWrapper, DiffusionNetDataset
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from shape_regression.regression_tools import RegressionTools, succinct_label_save_name, seed_all
from dataset.process_and_save_temp import Augmentation
import paths
from torch.utils.data import DataLoader
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from dataset.VertexAggregator import ThicknessThresholdAggregator

# system things
device = torch.device('cuda')
dtype = torch.float32

cache_operators = False

torch.cuda.empty_cache()

label_names = [
    # "Volume"
    "Thickness"
    # "curvature"
    # "SurfaceArea"
    # "Volume"
]
input_append_vertex_label_names = []
input_append_global_label_names = []

##### Classification
# category_thresholds = [0.05]
category_thresholds = None

num_outputs = len(label_names)
last_layer = None
if category_thresholds is not None:
    num_outputs = len(category_thresholds) + 1
    last_layer = "softmax"

######## Aggregation
# aggregator = ThicknessThresholdAggregator(warning_thickness=0.02, failure_thickness=0.05)
aggregator = None
aggregator_name = "none"
if aggregator is not None:
    aggregator_name = aggregator.get_name()

rot_augmentations = [Augmentation(scale=np.array([1.0, 1.0, 1.0]),
                                    rotation=np.array([np.pi, 0.0, 0.0])),
                        Augmentation(scale=np.array([1.0, 1.0, 1.0]),
                                      rotation=np.array([np.pi / 2, 0.0, 0.0])),
                        Augmentation(scale=np.array([1.0, 1.0, 1.0]),
                                      rotation=np.array([-np.pi / 2, 0.0, 0.0]))]
scale_augmentations = [Augmentation(scale=np.array([2.0, 1.0, 1.0]),
                                      rotation=np.array([0.0, 0.0, 0.0])),
                        Augmentation(scale=np.array([1.0, 2.0, 1.0]),
                                      rotation=np.array([0.0, 0.0, 0.0])),
                        Augmentation(scale=np.array([1.0, 1.0, 2.0]),
                                      rotation=np.array([0.0, 0.0, 0.0]))]
augmentations = scale_augmentations
augmentations_string = []
for aug in augmentations:
    augmentations_string.append(aug.as_string())

model_args = {
    "input_feature_type": 'xyz', # xyz, hks
    "k_eig": 64,
    "additional_dimensions": len(input_append_vertex_label_names) + len(input_append_global_label_names),
    "num_outputs": num_outputs,
    "C_width": 256,
    "N_block": 3,
    "last_layer": last_layer,
    "outputs_at": 'vertices',
    "mlp_hidden_dims": [256, 128],
    "dropout": True,
    "with_gradient_features": True,
    "with_gradient_rotations": True,
    "diffusion_method": 'spectral', #"implicit_dense", #'spectral',
    "data_parallel": False
    # "device": device
}

experiment_name = succinct_label_save_name(label_names) + aggregator_name
if category_thresholds is not None:
    experiment_name += "_CAT"


args = {
    "dataset_name": "DaVinci/train/",
    "exp_name": experiment_name,
    "label_names": label_names,
    "input_append_vertex_label_names": input_append_vertex_label_names,
    "input_append_global_label_names": input_append_global_label_names,
    "aggregator": aggregator_name,
    "seed": 0,
    "augmentations": "none",
    "remove_outlier_ratio": 0.0,
    "use_imbalanced_weights": False,

    # Dataset Param
    "data_fraction": 0.3,
    "num_data": None,
    "do_test": True,
    "workers": 1,
    "augment_random_rotate": (model_args["input_feature_type"] == 'xyz'),
    "use_category_thresholds": category_thresholds,

    # Opt Param
    "batch_size": 1,
    "test_batch_size": 1,
    "grad_acc_steps": 128,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-5,

    "scheduler": "plateau",
    # "restarts": 3,
    # "min_lr": 5e-4,
    "patience": 5,
    "factor": 0.1,

    "notes": "after_dataset_fix_no_weights"
}

args.update(model_args)

if __name__=="__main__":
    seed_all(args["seed"])
    data_root_dir = paths.CACHED_DATASETS_PATH + args["dataset_name"]
    op_cache_dir = data_root_dir + "../op_cache/"
    print(op_cache_dir)

    train_loader = DataLoader(DiffusionNetDataset(data_root_dir, model_args["k_eig"],
                                                  op_cache_dir=op_cache_dir,
                                                  partition="train",
                                                  num_data=args["num_data"],
                                                  data_fraction=args["data_fraction"], label_names=label_names,
                                                  augment_random_rotate=args["augment_random_rotate"],
                                                  extra_vertex_label_names=args["input_append_vertex_label_names"],
                                                  extra_global_label_names=args["input_append_global_label_names"],
                                                  outputs_at=args["outputs_at"],
                                                  use_imbalanced_weights=args["use_imbalanced_weights"],
                                                  augmentations=args["augmentations"],
                                                  remove_outlier_ratio=args["remove_outlier_ratio"],
                                                  cache_operators=cache_operators,
                                                  aggregator=aggregator,
                                                  categorical_thresholds=category_thresholds),
                                  num_workers=1,
                                  batch_size=args['batch_size'], shuffle=True, drop_last=True)
    test_loader = None
    if args["do_test"]:
        test_loader = DataLoader(DiffusionNetDataset(data_root_dir, model_args["k_eig"],
                                                     op_cache_dir=op_cache_dir,
                                                     partition="validation",
                                                     num_data=args["num_data"],
                                                     data_fraction=args["data_fraction"], label_names=label_names,
                                                     augment_random_rotate=args["augment_random_rotate"],
                                                     extra_vertex_label_names=args["input_append_vertex_label_names"],
                                                     extra_global_label_names=args["input_append_global_label_names"],
                                                     outputs_at=args["outputs_at"],
                                                     use_imbalanced_weights=args["use_imbalanced_weights"],
                                                     augmentations=args["augmentations"],
                                                     remove_outlier_ratio=args["remove_outlier_ratio"],
                                                     aggregator=aggregator,
                                                     cache_operators=cache_operators,
                                                     categorical_thresholds=category_thresholds),
                                 num_workers=1,
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

    regression_manager.train(args, do_test=args["do_test"], plot_every_n_epoch=5)






