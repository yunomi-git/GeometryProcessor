from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import paths
from dgcnn_net.data import ModelNet40
from dgcnn_net.model import DGCNN_XL, DGCNN
import numpy as np
from torch.utils.data import DataLoader
from dgcnn_net.util import cal_loss, IOStream
import sklearn.metrics as metrics
from tqdm import tqdm
import math
from heuristic_prediction.mesh_dataloader import DGCNNDataSet

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args['exp_name']):
        os.makedirs('checkpoints/' + args['exp_name'])
    if not os.path.exists('checkpoints/' + args['exp_name'] + '/' + 'models'):
        os.makedirs('checkpoints/' + args['exp_name'] + '/' + 'models')
    # os.system('cp main.py checkpoints' + '/' + args['exp_name'] + '/' + 'main.py.backup')
    # os.system('cp model.py checkpoints' + '/' + args['exp_name'] + '/' + 'model.py.backup')
    # os.system('cp util.py checkpoints' + '/' + args['exp_name'] + '/' + 'util.py.backup')
    # os.system('cp data.py checkpoints' + '/' + args['exp_name'] + '/' + 'data.py.backup')

def filter_criteria(mesh, instance_data) -> bool:
    # if mesh_data["vertices"] > 1e4:
    #     return true
    if instance_data["vertices"] < 1e3:
        return False
    # if instance_data["vertices"] > 1e5:
    #     return False
    if math.isnan(instance_data["thickness_violation"]):
        return False
    # if instance_data["scale"] > 1000:
    #     return False

    return True


def train(args, io):
    # train_loader = DataLoader(ModelNet40(partition='train', num_points=args["num_points"]), num_workers=8,
    #                           batch_size=args["batch_size"], shuffle=True, drop_last=True)
    # test_loader = DataLoader(ModelNet40(partition='test', num_points=args["num_points"]), num_workers=8,
    #                          batch_size=args["test_batch_size"], shuffle=True, drop_last=False)
    data_root_dir = paths.HOME_PATH + "data_augmentations/"

    train_loader = DataLoader(DGCNNDataSet(data_root_dir, args['num_points'], partition='train',
                                           filter_criteria=filter_criteria, use_augmentations=False),
                              num_workers=16,
                              batch_size=args['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(DGCNNDataSet(data_root_dir, args['num_points'], partition='test',
                                          filter_criteria=filter_criteria, use_augmentations=False),
                             num_workers=16,
                             batch_size=args['test_batch_size'], shuffle=True, drop_last=False)

    device = torch.device("cuda")

    model = DGCNN_XL(args).to(device)
    # print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    opt = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args['epochs'], eta_min=args['lr'])

    # criterion = cal_loss
    # This is for classification
    criterion = torch.nn.MSELoss()

    best_test_acc = 0
    for epoch in range(args['epochs']):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in tqdm(train_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f' % (epoch, train_loss * 1.0 / count)
        # outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
        #                                                                          train_loss * 1.0 / count,
        #                                                                          metrics.accuracy_score(train_true,
        #                                                                                                 train_pred),
        #                                                                          metrics.balanced_accuracy_score(
        #                                                                              train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        outstr = 'Train %d, loss: %.6f' % (epoch, train_loss * 1.0 / count)
        # test_acc = metrics.accuracy_score(test_true, test_pred)
        # avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        #
        # outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
        #                                                                       test_loss * 1.0 / count,
        #                                                                       test_acc,
        #                                                                       avg_per_class_acc)
        io.cprint(outstr)
        # if test_acc >= best_test_acc:
        #     best_test_acc = test_acc
        #     torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args['exp_name'])


def run_test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args['num_points']),
                             batch_size=args['test_batch_size'], shuffle=True, drop_last=False)

    device = torch.device("cuda")

    # Try to load models
    model = DGCNN_XL(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args['model_path']))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in tqdm(test_loader):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    args = {
        "exp_name": 'exp',
        "batch_size": 8,
        "test_batch_size": 8,
        "epochs": 200,
        "lr": 1e-3,
        "seed": 1,
        "eval": False,
        "num_points": 1024,
        "dropout": 0.5,
        "emb_dims": 1024,
        "k": 20,
        "num_outputs": 4,
        "model_path": "checkpoints/exp/models/"
    }

    _init_(args)

    io = IOStream('checkpoints/' + args['exp_name'] + '/run.log')

    torch.manual_seed(args['seed'])
    io.cprint(
        'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    torch.cuda.manual_seed(args['seed'])

    if not args['eval']:
        train(args, io)
    else:
        run_test(args, io)
