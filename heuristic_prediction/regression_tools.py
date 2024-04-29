import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics
import torch.backends.cudnn as cudnn
import os
from pathlib import Path
import json
import util
from util import IOStream, Stopwatch
import matplotlib.pyplot as plt
from datetime import datetime

class RegressionTools:
    def __init__(self, args, label_names, train_loader, test_loader, model, opt, scheduler=None, clip_parameters=False):
        self.device = torch.device("cuda")
        # self.seed_all(args["seed"])

        self.dataset_name = args["dataset_name"]
        self.model_name = type(model).__name__
        self.model = nn.DataParallel(model.to(self.device))
        self.label_names = label_names

        self.num_outputs = len(label_names)
        self.test_loader = test_loader
        self.train_loader = train_loader

        self.loss_criterion = torch.nn.MSELoss()

        self.clip_parameters = clip_parameters
        self.clip_threshold = 1

        cudnn.benchmark = True

        self.opt = opt
        self.scheduler = scheduler

        self.args = args
        self.checkpoint_path = ("checkpoints/" + self.dataset_name + "/" + self.model_name + "/" +
                                util.get_date_name() + "_" + args['exp_name'] + "/")

        print("Saving checkpoints to: ", self.checkpoint_path)
        self.io = self.create_checkpoints(args)
        self.gradient_accumulation_steps = args["grad_acc_steps"]
        # self.io.cprint("Total training data: " + str(self.train_loader.))


    def create_checkpoints(self, args):
        Path(self.checkpoint_path + 'models/').mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_path + 'images/').mkdir(parents=True, exist_ok=True)
        # save the arguments
        with open(self.checkpoint_path + "args.json", "w") as f:
            json.dump(args, f, indent=4)

        io = IOStream(self.checkpoint_path + 'run.log')
        return io



    def train(self, args, do_test=True, plot_every_n_epoch=-1, outputs_at="global"):
        best_res_mag = 0
        loss_history = []
        res_train_history = []
        labels = args["label_names"]
        for epoch in range(args['epochs']):
            print(datetime.now())
            ####################
            # Train
            ####################
            train_loss = 0.0
            count = 0.0
            self.model.train()
            self.opt.zero_grad()
            train_pred = []
            train_true = []
            with torch.enable_grad():
                for batch_idx, (data, label, weight) in enumerate(tqdm(self.train_loader)):
                    data, label = data.to(self.device), label.to(self.device)
                    weight = weight.to(self.device)
                    weight = torch.sqrt(weight)

                    preds = self.model(data)
                    loss = self.loss_criterion(weight * preds, weight * label) / self.gradient_accumulation_steps
                    loss.backward()

                    if batch_idx % self.gradient_accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                        if self.clip_parameters:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_threshold)
                        self.opt.step()
                        self.opt.zero_grad()

                    batch_size = data.size()[0]
                    count += batch_size
                    train_loss += loss.item() * batch_size
                    train_true.append(label.cpu().numpy())
                    train_pred.append(preds.detach().cpu().numpy())

            if self.scheduler is not None:
                if self.args["scheduler"] == "plateau":
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)

            # for vertices, outputs as batch x dim x points. swap to batch*points x dim
            if outputs_at == "vertices":
                train_true = train_true.reshape(
                    (train_true.shape[0] * train_true.shape[1], train_true.shape[2]),
                    order="F")
                train_pred = train_pred.reshape(
                    (train_pred.shape[0] * train_pred.shape[1], train_pred.shape[2]),
                    order="F")

            res = metrics.r2_score(y_true=train_true, y_pred=train_pred, multioutput='raw_values') #num_val x dims
            acc = get_accuracy_tolerance(preds=train_pred, actual=train_true, tolerance=0.05)

            outstr = ('========\nTrain Epoch: %d '
                      '\n\tLoss: %.6f '
                      '\n\tr2: %s '
                      '\n\tlr: %s '
                      '\n\tacc: %s' %
                      (epoch,
                       train_loss * 1.0 / count,
                       str(res),
                       self.opt.param_groups[0]['lr'],
                       str(acc)))
            self.io.cprint(outstr)

            loss_history.append(train_loss * 1.0 / count)
            res_train_history.append(res)

            # Loss
            if (plot_every_n_epoch >= 1 and epoch % plot_every_n_epoch == 0) or epoch + 1 == args['epochs']:
                timer = Stopwatch()
                timer.start()
                plt.figure(0)
                plt.clf()
                for i in range(len(res)):
                    plt.subplot(len(res), 1, i+1)
                    plt.scatter(train_true[:, i], train_pred[:, i] - train_true[:, i], alpha=0.15)
                    plt.hlines(y=0, xmin=min(train_true[:, i]), xmax=max(train_true[:, i]), color="red")
                    plt.xlabel("Train True")
                    plt.ylabel(labels[i] + " Error")

                plt.savefig(self.checkpoint_path + "images/" + 'confusion_' + str(epoch) + '.png')

                plt.figure(2)
                plt.clf()

                for i in range(len(res)):
                    res_history = np.array(res_train_history)
                    plt.subplot(len(res), 1, i+1)
                    plt.plot(res_history[:, i])
                    plt.ylabel(labels[i] + " Train R2")
                    plt.xlabel("Epoch")

                plt.savefig(self.checkpoint_path + "images/" + 'r2_train.png')

                plt.figure(1)
                plt.clf()
                plt.plot(loss_history)
                plt.ylabel("Loss")
                plt.xlabel("Epoch")
                plt.savefig(self.checkpoint_path + "images/" + 'loss.png')
                print("Time to save figures: ", timer.get_time())



            ####################
            # Test
            ####################
            if do_test:
                test_loss = 0.0
                count = 0.0
                self.model.eval()
                test_pred = []
                test_true = []
                with torch.no_grad():
                    for data, label, weight in tqdm(self.test_loader):
                        data, label = data.to(self.device), label.to(self.device)
                        weight = weight.to(self.device)
                        weight = torch.sqrt(weight)
                        # data = data.permute(0, 2, 1)
                        preds = self.model(data)
                        loss = self.loss_criterion(weight * preds, weight * label)

                        batch_size = data.size()[0]
                        count += batch_size
                        test_loss += loss.item() * batch_size
                        test_true.append(label.cpu().numpy())
                        test_pred.append(preds.detach().cpu().numpy())

                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)

                if outputs_at == "vertices":
                    test_true = test_true.reshape(
                        (test_true.shape[0] * test_true.shape[1], test_true.shape[2]),
                        order="F")
                    test_pred = test_pred.reshape(
                        (test_pred.shape[0] * test_pred.shape[1], test_pred.shape[2]),
                        order="F")

                res = metrics.r2_score(y_true=test_true, y_pred=test_pred, multioutput='raw_values')
                outstr = '--\n\tTest: \n\t\tLoss: %.6f \n\t\tr2: %s\n=========' % (test_loss * 1.0 / count, str(res))
                self.io.cprint(outstr)

                if (res > 0).all() and np.linalg.norm(res) > best_res_mag:
                    best_res_mag = np.linalg.norm(res)
                    torch.save(self.model.state_dict(), self.checkpoint_path + 'model.t7')
            else:
                if (res > 0).all() and np.linalg.norm(res) > best_res_mag:
                    best_res_mag = np.linalg.norm(res)
                    torch.save(self.model.state_dict(), self.checkpoint_path + 'model.t7')

def get_accuracy_tolerance(preds: np.ndarray, actual: np.ndarray, tolerance=0.1):
    # input preds, actual: N x M for N values, M labels
    # output: 1xM accuracy per label
    within_tolerance = np.abs(preds - actual) <= tolerance
    accuracy = np.count_nonzero(within_tolerance, axis=0) / len(preds)
    return accuracy


def succinct_label_save_name(label_names):
    out = ""
    for name in label_names:
        out += name[:5] + "-"
    return out

def seed_all(seed):
    torch.cuda.empty_cache()

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed) # What is this?
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # What is this?
# class TrainingLogger:
#     def __init__(self, save_path):
#         self.save_path = save_path
#
#     def log_current_epoch(self):
#         loss_history.append(train_loss * 1.0 / count)
#         res_train_history.append(res)
#
#     def print(self):
#         outstr = ('========\nTrain Epoch: %d '
#                   '\n\tLoss: %.6f '
#                   '\n\tr2: %s '
#                   '\n\tlr: %s '
#                   '\n\tacc: %s' %
#                   (epoch, train_loss * 1.0 / count, str(res), self.opt.param_groups[0]['lr'], str(acc)))
#         self.io.cprint(outstr)
#
#     def save_images(self):
#         plt.figure(0)
#         plt.clf()
#         plt.scatter(train_true, train_pred)
#         plt.xlabel("Train True")
#         plt.ylabel("Train Pred")
#         plt.savefig(self.checkpoint_path + "images/" + 'confusion_' + str(epoch) + '.png')
#
#         plt.figure(1)
#         plt.clf()
#         plt.plot(loss_history)
#         plt.ylabel("Loss")
#         plt.xlabel("Epoch")
#         plt.savefig(self.checkpoint_path + "images/" + 'loss.png')
#
#         plt.figure(2)
#         plt.clf()
#         plt.plot(res_train_history)
#         plt.ylabel("Train R2")
#         plt.xlabel("Epoch")
#         plt.savefig(self.checkpoint_path + "images/" + 'r2_train.png')
