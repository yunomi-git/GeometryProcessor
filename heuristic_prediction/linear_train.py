import torch
import torch.nn as nn
from regression_tools import RegressionTools
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from heuristic_prediction.regression_tools import RegressionTools, succinct_label_save_name
import paths
from dgcnn_net.model import DGCNN_param, DGCNN_XL, DGCNN
from torch.utils.data import DataLoader
import math
from heuristic_prediction.pointcloud_dataloader import load_point_clouds_numpy
import numpy as np
from torch.utils.data import Dataset

torch.cuda.empty_cache()

label_names = [
    "centroid_x"
]

model_args = {
    "num_points": 2048,
    "conv_channel_sizes": [128, 128, 256, 512],  # Default: [64, 64, 128, 256] #Mo: [512, 512, 1024]
    "emb_dims": 256,
    "linear_sizes": [1024, 512, 256, 128, 64, 32, 16],  # [512, 256] #Mo: [1024, 512, 256, 128, 64, 32, 16]
    "num_outputs": len(label_names),
    "k": 10,
    "dropout": 0.2,
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
    "normalize_inputs": True,
    "sampling_method": "mixed",

    # Opt Param
    "batch_size": 16,
    "test_batch_size": 8,
    "epochs": 30,
    "lr": 1e-2,
    "min_lr": 5e-5,
    "weight_decay": 1e-4,
    "seed": 1,
}

args.update(model_args)

class LinearModel(nn.Module):
    def __init__(self, args: dict):
        super(LinearModel, self).__init__()
        self.linear_layer = nn.Linear(3, 3)


    def forward(self, x):
        # batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        x = self.linear_layer(x)
        mean = torch.mean(x, dim=1)
        mean = torch.mean(mean, dim=1)

        return mean

class CentroidDataset(Dataset):
    def __init__(self, data_root_dir, num_points, label_names, partition='train',
                 data_fraction=1.0, normalize=False, sampling_method="mixed"):
        print("Loading data from numpy...")
        if normalize:
            print("Note: Data is being normalized")
        self.point_clouds, self.label = load_point_clouds_numpy(data_root_dir=data_root_dir,
                                                                num_points=2 * num_points,
                                                                label_names=label_names,
                                                                data_fraction=data_fraction,
                                                                sampling_method=sampling_method)

        # Normalize each target
        if normalize:
            # std = np.std(self.label, axis=0)
            # mean = np.mean(self.label, axis=0)
            # self.label = (self.label - mean) / std

            # Normalize inputs
            # First get bounding boxes
            bound_max = np.max(self.point_clouds, axis=1)
            bound_min = np.min(self.point_clouds, axis=1)
            bound_length = np.max(bound_max - bound_min, axis=1) # maximum length
            # scale to box of 0, 1
            self.normalization_scale = 1.0/bound_length
            normalization_scale_multiplier = np.repeat(self.normalization_scale[:, np.newaxis], len(self.point_clouds[0]), axis=1)
            normalization_scale_multiplier = np.repeat(normalization_scale_multiplier[:, :, np.newaxis], 3, axis=2)
            centering = np.repeat(bound_min[:, np.newaxis, :], num_points*2, axis=1)
            self.normalization_order = 1
            self.point_clouds -= centering
            self.point_clouds = normalization_scale_multiplier * self.point_clouds
            # self.label = self.label * np.repeat(np.power(self.normalization_scale, self.normalization_order)[:, np.newaxis], len(self.label[0]), axis=1)


        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.point_clouds[item][:self.num_points] # Sample points from the mesh
        label = np.mean(pointcloud, axis=0)[0]
        label = np.array([label])
        # label = self.label[item] # Grab the output
        if self.partition == 'train':
            # pointcloud = dgcnn_data.translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.point_clouds.shape[0]

if __name__=="__main__":
    ### Data ###
    data_root_dir = paths.CACHED_DATASETS_PATH + "data_th5k_aug/"  # "data_augmentations/" #
    train_loader = DataLoader(CentroidDataset(data_root_dir, args['num_points'], label_names=label_names,
                                                partition='train',
                                                data_fraction=args["data_fraction"],
                                                normalize=args["normalize_inputs"],
                                                sampling_method=args["sampling_method"]),
                              num_workers=24,
                              batch_size=args['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(CentroidDataset(data_root_dir, args['num_points'], label_names=label_names,
                                               partition='test',
                                               data_fraction=args["data_fraction_test"],
                                               normalize=args["normalize_inputs"],
                                               sampling_method=args["sampling_method"]),
                             num_workers=24,
                             batch_size=args['test_batch_size'], shuffle=True, drop_last=False)

    ### Model ###
    model = DGCNN_param(args)

    # opt = torch.optim.SGD(model.parameters(), lr=args["lr"], momentum=0.9, weight_decay=args["weight_decay"])
    opt = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args["weight_decay"])
    # scheduler = CosineAnnealingLR(opt, args["epochs"], eta_min=args["min_lr"])
    scheduler = None

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