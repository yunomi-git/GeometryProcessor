from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import paths
import util
from dgcnn_net.data import ModelNet40
from dgcnn_net.model import DGCNN_param, DGCNN
import numpy as np
from torch.utils.data import DataLoader
from dgcnn_net.util import cal_loss, IOStream
import sklearn.metrics as metrics
from tqdm import tqdm
import math
from heuristic_prediction.mesh_dataloader import PointCloudDataset
from lightning.fabric import Fabric
import argparse
import sys

# def parse_args():
#     """Parameters"""
#     parser = argparse.ArgumentParser('training')
#     parser.add_argument('-r', '--rank', type=int)
#     return parser.parse_args()
#
# pargs = parse_args()
torch.cuda.empty_cache()
fabric = Fabric(accelerator="cuda", devices=2)
fabric.launch()


# def _init_(args):
#     if not os.path.exists('checkpoints'):
#         os.makedirs('checkpoints')
#     if not os.path.exists('checkpoints/' + args['exp_name']):
#         os.makedirs('checkpoints/' + args['exp_name'])
#     if not os.path.exists('checkpoints/' + args['exp_name'] + '/' + 'models'):
#         os.makedirs('checkpoints/' + args['exp_name'] + '/' + 'models')

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


label_names = [
    # "overhang_violation",
    # "stairstep_violation",
    "thickness_violation",
    "gap_violation"
]

def train(args, io=None):
    data_root_dir = paths.HOME_PATH + "data_augmentations/"

    train_loader = DataLoader(PointCloudDataset(data_root_dir, args['num_points'], label_names=label_names, partition='train',
                                                filter_criteria=filter_criteria, use_augmentations=True,
                                                data_fraction=args["data_fraction"]),
                              num_workers=16,
                              batch_size=args['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(PointCloudDataset(data_root_dir, args['num_points'], label_names=label_names, partition='test',
                                               filter_criteria=filter_criteria, use_augmentations=False,
                                               data_fraction=args["data_fraction"]),
                             num_workers=16,
                             batch_size=args['test_batch_size'], shuffle=True, drop_last=False)

    device = torch.device("cuda")

    model = DGCNN_XL(args)#.to(device)
    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    opt = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)

    # Setup Fabric
    model, opt = fabric.setup(model, opt)
    train_loader = fabric.setup_dataloaders(train_loader)
    test_loader = fabric.setup_dataloaders(test_loader)

    scheduler = CosineAnnealingLR(opt, args['epochs'], eta_min=args['lr'])

    criterion = torch.nn.MSELoss()

    best_res_mag = 0

    for epoch in range(args['epochs']):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        with torch.enable_grad():
            for data, label in tqdm(train_loader):
                # data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                opt.zero_grad()
                logits = model(data)
                loss = criterion(logits, label)
                fabric.backward(loss)
                # loss.backward()
                opt.step()
                # preds = logits.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                train_true.append(label.cpu().numpy())
                train_pred.append(logits.detach().cpu().numpy())
        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        res = metrics.r2_score(y_true=train_true, y_pred=train_pred, multioutput='raw_values')
        outstr = 'Train %d, loss: %.6f' % (epoch, train_loss * 1.0 / count)
        print(outstr, "residual: " + str(res))
        # io.cprint(outstr)
        # io.cprint("residual: " + str(res))

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        with torch.no_grad():
            for data, label in tqdm(test_loader):
                # data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits = model(data)
                loss = criterion(logits, label)
                # preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(logits.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        res = metrics.r2_score(y_true=test_true, y_pred=test_pred, multioutput='raw_values')
        outstr = 'Train %d, loss: %.6f' % (epoch, test_loss * 1.0 / count)
        print(outstr, "residual: " + str(res))

        # io.cprint(outstr)
        # io.cprint("test residual: " + str(res))
        # if (res > 0).all() and np.linalg.norm(res) > best_res_mag:
        #     best_res_mag = np.linalg.norm(res)
        #     torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args['exp_name'])


def run_test(args, io=None):
    data_root_dir = paths.HOME_PATH + "data_augmentations/"
    test_loader = DataLoader(PointCloudDataset(data_root_dir, args['num_points'], label_names=label_names, partition='test',
                                               filter_criteria=filter_criteria, use_augmentations=False),
                             num_workers=16,
                             batch_size=args['test_batch_size'], shuffle=True, drop_last=False)

    device = torch.device("cuda")

    # Try to load models
    model = DGCNN_XL(args)#.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args['model_path']))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in tqdm(test_loader):
        # data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    # test_acc = metrics.accuracy_score(test_true, test_pred)
    # avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    # outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    # io.cprint(outstr)


if __name__ == "__main__":
    # my_task_id = int(sys.argv[1])
    # num_tasks = int(sys.argv[2])

    name = "gapthick" + "_" + util.get_date_name()
    args = {
        "data_fraction": 1.0,
        "exp_name": name,

        # Opt Param
        "batch_size": 4,
        "test_batch_size": 4,
        "epochs": 100,
        "lr": 1e-3,
        "seed": 1,
        "eval": False,

        "model_path": "checkpoints/" + name + "/models/",

        # Model Parameters
        "num_points": 4096,
        "conv_channel_sizes": [64, 64, 128, 256], # [64, 64, 128, 256]
        "emb_dims": 512,
        "linear_sizes": [512, 256], # [512, 256]
        "num_outputs": 2,
        "k": 40,
        "dropout": 0.2,
    }

    # if my_task_id == 0:
    #     _init_(args)

    # io = IOStream('checkpoints/' + args['exp_name'] + '/run.log')
    io=None

    torch.manual_seed(args['seed'])
    # io.cprint(
    #     'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    torch.cuda.manual_seed(args['seed'])

    if not args['eval']:
        train(args, io)
    else:
        run_test(args, io)
