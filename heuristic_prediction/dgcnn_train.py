from __future__ import print_function
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from heuristic_prediction.regression_tools import RegressionTools
import paths
from dgcnn_net.model import DGCNN_param, DGCNN_XL, DGCNN
from torch.utils.data import DataLoader
import math
from heuristic_prediction.mesh_dataloader import PointCloudDataset

from dgcnn_net.util import cal_loss, IOStream
import sklearn.metrics as metrics
from tqdm import tqdm
import util
from dgcnn_net.data import ModelNet40
import numpy as np
import torch.nn as nn
import os

torch.cuda.empty_cache()

experiment_name = "gapthick"

label_names = [
    # "overhang_violation",
    # "stairstep_violation",
    # "thickness_violation",
    # "gap_violation",
    # "volume",
    # "bound_length",
    # "surface_area"
    "centroid_x"
]

model_args = {
    "num_points": 2048,
    "conv_channel_sizes": [128, 128, 256, 512],  # [64, 64, 128, 256] #XL: 256, 256, 512, 1024, 2048
    "emb_dims": 512,
    "linear_sizes": [2048, 1024, 512],  # [512, 256] #XL: 4096, 2048, 1024, 512
    "num_outputs": len(label_names),
    "k": 40,
    "dropout": 0.2,
}

args = {
    "exp_name": experiment_name,
    "label_names": label_names,

    # Dataset Param
    "data_fraction": 0.3,
    "data_fraction_test": 0.15,
    "workers": 24,

    # Opt Param
    "batch_size": 4,
    "test_batch_size": 8,
    "epochs": 100,
    "lr": 1e-1,
    "min_lr": 5e-3,
    "weight_decay": 1e-4,
    "seed": 1,
}

args.update(model_args)



def filter_criteria(mesh, instance_data) -> bool:
    # if mesh_data["vertices"] > 1e4:
    #     return true
    # if instance_data["vertices"] < 1e3:
    #     return False
    # if instance_data["vertices"] > 1e5:
    #     return False
    if math.isnan(instance_data["thickness_violation"]):
        return False
    # if instance_data["scale"] > 1000:
    #     return False

    return True

if __name__ == "__main__":
    ### Data ###
    data_root_dir = paths.HOME_PATH + "data_th5k_aug/"# "data_augmentations/" #
    train_loader = DataLoader(PointCloudDataset(data_root_dir, args['num_points'], label_names=label_names, partition='train',
                                                filter_criteria=filter_criteria, use_augmentations=True,
                                                data_fraction=args["data_fraction"], use_numpy=True),
                              num_workers=24,
                              batch_size=args['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(PointCloudDataset(data_root_dir, args['num_points'], label_names=label_names, partition='test',
                                               filter_criteria=filter_criteria, use_augmentations=False,
                                               data_fraction=args["data_fraction_test"], use_numpy=True),
                             num_workers=24,
                             batch_size=args['test_batch_size'], shuffle=True, drop_last=False)

    ### Model ###
    model = DGCNN_param(args)

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

    regression_manager.train(args, do_test=False)

# def _init_(args):
#     if not os.path.exists('checkpoints'):
#         os.makedirs('checkpoints')
#     if not os.path.exists('checkpoints/' + args['exp_name']):
#         os.makedirs('checkpoints/' + args['exp_name'])
#     if not os.path.exists('checkpoints/' + args['exp_name'] + '/' + 'models'):
#         os.makedirs('checkpoints/' + args['exp_name'] + '/' + 'models')
#     # os.system('cp main.py checkpoints' + '/' + args['exp_name'] + '/' + 'main.py.backup')
#     # os.system('cp model.py checkpoints' + '/' + args['exp_name'] + '/' + 'model.py.backup')
#     # os.system('cp util.py checkpoints' + '/' + args['exp_name'] + '/' + 'util.py.backup')
#     # os.system('cp data.py checkpoints' + '/' + args['exp_name'] + '/' + 'data.py.backup')

#
# def train(args, io):
#     data_root_dir = paths.HOME_PATH + "data_th5k_aug/"# "data_augmentations/" #
#
#     train_loader = DataLoader(PointCloudDataset(data_root_dir, args['num_points'], label_names=label_names, partition='train',
#                                                 filter_criteria=filter_criteria, use_augmentations=True,
#                                                 data_fraction=args["data_fraction"], use_numpy=True),
#                               num_workers=24,
#                               batch_size=args['batch_size'], shuffle=True, drop_last=True)
#     test_loader = DataLoader(PointCloudDataset(data_root_dir, args['num_points'], label_names=label_names, partition='test',
#                                                filter_criteria=filter_criteria, use_augmentations=False,
#                                                data_fraction=args["data_fraction_test"], use_numpy=True),
#                              num_workers=24,
#                              batch_size=args['test_batch_size'], shuffle=True, drop_last=False)
#
#     device = torch.device("cuda")
#
#     model = DGCNN_param(args).to(device)
#     # print(str(model))
#
#     model = nn.DataParallel(model)
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#
#     opt = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)
#
#     scheduler = CosineAnnealingLR(opt, args['epochs'], eta_min=args['lr'])
#
#     # criterion = cal_loss
#     # This is for classification
#     criterion = torch.nn.MSELoss()
#
#     best_test_acc = 0
#     best_res_mag = 0
#
#     for epoch in range(args['epochs']):
#         ####################
#         # Train
#         ####################
#         train_loss = 0.0
#         count = 0.0
#         model.train()
#         train_pred = []
#         train_true = []
#         with torch.enable_grad():
#             for data, label in tqdm(train_loader):
#                 data, label = data.to(device), label.to(device)
#                 data = data.permute(0, 2, 1)
#                 batch_size = data.size()[0]
#                 opt.zero_grad()
#                 logits = model(data)
#                 loss = criterion(logits, label)
#                 loss.backward()
#                 opt.step()
#                 count += batch_size
#                 train_loss += loss.item() * batch_size
#                 train_true.append(label.cpu().numpy())
#                 train_pred.append(logits.detach().cpu().numpy())
#         scheduler.step()
#         train_true = np.concatenate(train_true)
#         train_pred = np.concatenate(train_pred)
#         res = metrics.r2_score(y_true=train_true, y_pred=train_pred, multioutput='raw_values')
#         outstr = 'Train %d, loss: %.6f' % (epoch, train_loss * 1.0 / count)
#         io.cprint(outstr)
#         io.cprint("residual: " + str(res))
#
#         ####################
#         # Test
#         ####################
#         test_loss = 0.0
#         count = 0.0
#         model.eval()
#         test_pred = []
#         test_true = []
#         with torch.no_grad():
#             for data, label in tqdm(test_loader):
#                 data, label = data.to(device), label.to(device)
#                 data = data.permute(0, 2, 1)
#                 batch_size = data.size()[0]
#                 logits = model(data)
#                 loss = criterion(logits, label)
#                 count += batch_size
#                 test_loss += loss.item() * batch_size
#                 test_true.append(label.cpu().numpy())
#                 test_pred.append(logits.detach().cpu().numpy())
#
#         test_true = np.concatenate(test_true)
#         test_pred = np.concatenate(test_pred)
#         res = metrics.r2_score(y_true=test_true, y_pred=test_pred, multioutput='raw_values')
#         outstr = 'Train %d, loss: %.6f' % (epoch, test_loss * 1.0 / count)
#         io.cprint(outstr)
#         io.cprint("test residual: " + str(res))
#         if (res > 0).all() and np.linalg.norm(res) > best_res_mag:
#             best_res_mag = np.linalg.norm(res)
#             torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args['exp_name'])



# if __name__ == "__main__":
#     name = "volbb" + "_" + util.get_date_name()
#     args = {
#         "data_fraction": 0.3,
#         "data_fraction_test": 0.15,
#         "exp_name": name,
#
#         # Opt Param
#         "batch_size": 4,
#         "test_batch_size": 8,
#         "epochs": 100,
#         "lr": 1e-1,
#         "seed": 1,
#         "eval": False,
#
#         "model_path": "checkpoints/" + name + "/models/",
#
#         # Model Parameters
#         "num_points": 2048,
#         "conv_channel_sizes": [128, 128, 256, 512], # [64, 64, 128, 256] #XL: 256, 256, 512, 1024, 2048
#         "emb_dims": 512,
#         "linear_sizes": [2048, 1024, 512], # [512, 256] #XL: 4096, 2048, 1024, 512
#         "num_outputs": len(label_names),
#         "k": 40,
#         "dropout": 0.2,
#     }
#
#     _init_(args)
#
#     io = IOStream('checkpoints/' + args['exp_name'] + '/run.log')
#
#     torch.manual_seed(args['seed'])
#     io.cprint(
#         'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
#     torch.cuda.manual_seed(args['seed'])
#
#     train(args, io)


