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
    def __init__(self, args, label_names, train_loader, test_loader, model, opt, scheduler=None, clip_parameters=False,
                 include_faces=False):
        torch.cuda.empty_cache()
        self.device = torch.device("cuda")
        # self.seed_all(args["seed"])

        self.dataset_name = args["dataset_name"]
        self.model_name = type(model).__name__

        if "data_parallel" in args and not args["data_parallel"]:
            self.model = model.to(self.device)
        else:
            self.model = nn.DataParallel(model.to(self.device))
        # self.model = model.to(self.device)

        self.label_names = label_names

        self.num_outputs = len(label_names)
        self.test_loader = test_loader
        self.train_loader = train_loader

        self.loss_criterion = torch.nn.MSELoss()

        self.clip_parameters = clip_parameters
        self.clip_threshold = 1

        self.include_faces = include_faces

        cudnn.benchmark = True

        self.opt = opt
        self.scheduler = scheduler

        self.args = args
        self.checkpoint_path = ("checkpoints/" + self.dataset_name + "/" + self.model_name + "/" +
                                util.get_date_name() + "_" + args['exp_name'] + "/")

        print("Saving checkpoints to: ", self.checkpoint_path)
        self.io = self.create_checkpoints(args)
        self.gradient_accumulation_steps = args["grad_acc_steps"]
        self.NUM_PLOT_POINTS = 7500
        # self.io.cprint("Total training data: " + str(self.train_loader.))


    def create_checkpoints(self, args):
        Path(self.checkpoint_path + 'models/').mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_path + 'images/').mkdir(parents=True, exist_ok=True)
        # save the arguments
        with open(self.checkpoint_path + "args.json", "w") as f:
            json.dump(args, f, indent=4)

        io = IOStream(self.checkpoint_path + 'run.log')
        return io

    def get_evaluations_pointcloud(self, training=True):
        train_loss = 0.0
        count = 0.0
        self.model.train()
        self.opt.zero_grad()
        train_pred = []
        train_true = []
        if training:
            data_loader = self.train_loader
        else:
            data_loader = self.test_loader
        for batch_idx, (data, label, weight) in enumerate(tqdm(data_loader)):
            data, label = data.to(self.device), label.to(self.device)
            weight = weight.to(self.device)
            weight = torch.sqrt(weight)

            preds = self.model(data)
            if training:
                loss = self.loss_criterion(weight * preds, weight * label) / self.gradient_accumulation_steps
                loss.backward()

                if batch_idx % self.gradient_accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                    if self.clip_parameters:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_threshold)
                    self.opt.step()
                    self.opt.zero_grad()
            else:
                loss = self.loss_criterion(weight * preds, weight * label)

                if self.clip_parameters:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_threshold)
                self.opt.step()
                self.opt.zero_grad()

            batch_size = data.size()[0]
            count += batch_size
            train_loss += loss.item() * batch_size
            # train_true.append(label.cpu().numpy())
            # train_pred.append(preds.detach().cpu().numpy())

            label = label.cpu().numpy()
            preds = preds.detach().cpu().numpy()
            # for vertices, outputs as batch x dim x points. swap to batch*points x dim
            if len(label.shape) == 3:
                label = label.reshape((label.shape[0] * label.shape[1], label.shape[2]), order="F")
                preds = preds.reshape((preds.shape[0] * preds.shape[1], preds.shape[2]), order="F")

            train_true.append(label)
            train_pred.append(preds)

        return train_loss, count, train_pred, train_true

    def get_evaluations_mesh(self, training=True):
        train_loss = 0.0
        count = 0.0
        if training:
            self.model.train()
        else:
            self.model.eval()

        self.opt.zero_grad()
        train_pred = []
        train_true = []
        if training:
            data_loader = self.train_loader
        else:
            data_loader = self.test_loader
        for batch_idx, (vertices, faces, label) in enumerate(tqdm(data_loader)):
            vertices, faces, label = vertices.to(self.device), faces.to(self.device), label.to(self.device)
            # weight = weight.to(self.device)
            # weight = torch.sqrt(weight)
            weight = 1

            try:
                preds = self.model(vertices, faces)
            except Exception as e:
                print("Training: Error calculating model. Skipping")
                print(e)
                continue

            if training:
                loss = self.loss_criterion(weight * preds, weight * label) / self.gradient_accumulation_steps
                loss.backward()

                if batch_idx % self.gradient_accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                    if self.clip_parameters:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_threshold)
                    self.opt.step()
                    self.opt.zero_grad()
            else:
                loss = self.loss_criterion(weight * preds, weight * label)
                # TODO why is the following getting calculated?
                # loss.backward()

                if self.clip_parameters:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_threshold)
                self.opt.step()
                self.opt.zero_grad()

            batch_size = vertices.size()[0]
            count += batch_size
            train_loss += loss.item() * batch_size

            label = label.cpu().numpy()
            preds = preds.detach().cpu().numpy()

            if np.isnan(preds).any():
                print("nan detected. skipping batch")
                continue

            # for vertices, outputs as batch x dim x points. swap to batch*points x dim
            if len(label.shape) == 3:
                label = label.reshape((label.shape[0] * label.shape[1], label.shape[2]), order="F")
                preds = preds.reshape((preds.shape[0] * preds.shape[1], preds.shape[2]), order="F")

            train_true.append(label)
            train_pred.append(preds)

        return train_loss, count, train_pred, train_true

    def train(self, args, do_test=True, plot_every_n_epoch=-1):
        best_res_mag = 0
        loss_train_history = []
        res_train_history = []
        loss_test_history = []
        res_test_history = []
        labels = args["label_names"]
        for epoch in range(args['epochs']):
            print(datetime.now())
            ####################
            # Train
            ####################
            self.model.train()
            self.opt.zero_grad()

            with torch.enable_grad():
                if not self.include_faces:
                    train_loss, count, train_pred, train_true = self.get_evaluations_pointcloud(training=True)
                else:
                    train_loss, count, train_pred, train_true = self.get_evaluations_mesh(training=True)

            train_true = np.concatenate(train_true)
            # train_true = np.nan_to_num(train_true) # TODO this may not be the best option
            train_pred = np.concatenate(train_pred)

            if self.scheduler is not None:
                if self.args["scheduler"] == "plateau":
                    self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()

            res_train = metrics.r2_score(y_true=train_true, y_pred=train_pred, multioutput='raw_values') #num_val x dims
            acc = get_accuracy_tolerance(preds=train_pred, actual=train_true, tolerance=0.05)

            outstr = ('========\nTrain Epoch: %d '
                      '\n\tLoss: %.6f '
                      '\n\tr2: %s '
                      '\n\tlr: %s '
                      '\n\tacc: %s' %
                      (epoch,
                       train_loss * 1.0 / count,
                       str(res_train),
                       self.opt.param_groups[0]['lr'],
                       str(acc)))
            self.io.cprint(outstr)

            loss_train_history.append(train_loss * 1.0 / count)
            res_train_history.append(res_train)

            ####################
            # Test
            ####################
            if do_test:
                self.model.eval()
                with torch.no_grad():
                    if not self.include_faces:
                        test_loss, count, test_pred, test_true = self.get_evaluations_pointcloud(
                            training=False)
                    else:
                        test_loss, count, test_pred, test_true = self.get_evaluations_mesh(training=False)

                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)

                res_test = metrics.r2_score(y_true=test_true, y_pred=test_pred, multioutput='raw_values')
                outstr = ('--\n\tTest: \n\t\tLoss: %.6f \n\t\tr2: %s\n=========' %
                          (test_loss * 1.0 / count, str(res_test)))
                self.io.cprint(outstr)


                loss_test_history.append(train_loss * 1.0 / count)
                res_test_history.append(res_test)

            # Logging
            if (plot_every_n_epoch >= 1 and epoch % plot_every_n_epoch == 0) or epoch + 1 == args['epochs']:
                timer = Stopwatch()
                timer.start()
                self.save_r2(epoch, r2=res_train, true=train_true, pred=train_pred, labels=labels)

                if do_test:
                    self.save_r2(epoch, r2=res_test, true=test_true, pred=test_pred, labels=labels)
                    self.save_loss(labels=labels,
                                   res_train_history=res_train_history, loss_train_history=loss_train_history,
                                   res_test_history=res_test_history, loss_test_history=loss_test_history)
                else:
                    self.save_loss(labels=labels,
                                   res_train_history=res_train_history, loss_train_history=loss_train_history)

                print("Time to save figures: ", timer.get_time())

            # Save Checkpoint
            if do_test:
                res_mag = np.linalg.norm(res_test)
                if (res_test > 0).all() and res_mag > best_res_mag:
                    best_res_mag = res_mag
                    torch.save(self.model.state_dict(), self.checkpoint_path + 'model.t7')
            else:
                res_mag = np.linalg.norm(res_train)
                if (res_train > 0).all() and res_mag > best_res_mag:
                    best_res_mag = res_mag
                    torch.save(self.model.state_dict(), self.checkpoint_path + 'model.t7')


    def save_r2(self, epoch, r2, true, pred, labels):
        plt.figure(0)
        plt.clf()
        for i in range(len(r2)):
            plt.subplot(len(r2), 1, i + 1)
            error = pred[:, i] - true[:, i]
            true = true[:, i]

            plot_indices = np.arange(start=0, stop=len(error), step=int(np.ceil(len(error) / self.NUM_PLOT_POINTS)),
                                     dtype=np.int64)

            plt.scatter(true[plot_indices], error[plot_indices], alpha=0.15)
            plt.hlines(y=0, xmin=min(true), xmax=max(true), color="red")
            plt.xlabel("Train True")
            plt.ylabel(labels[i] + " Error")

        plt.savefig(self.checkpoint_path + "images/" + 'confusion_' + str(epoch) + '.png')

    def save_loss(self, labels, res_train_history, loss_train_history, res_test_history=None, loss_test_history=None):
        timer = Stopwatch()
        timer.start()

        plt.figure(2)
        plt.clf()

        for i in range(len(labels)):
            plt.subplot(len(labels), 1, i + 1)
            train_res_history = np.array(res_train_history)
            plt.plot(train_res_history[:, i], label='train')

            if res_test_history is not None:
                test_res_history = np.array(res_test_history)
                plt.plot(test_res_history[:, i], label='test')

            plt.ylabel(labels[i] + " Train R2")
            plt.xlabel("Epoch")
            plt.legend()

        plt.savefig(self.checkpoint_path + "images/" + 'r2_train.png')

        plt.figure(1)
        plt.clf()
        plt.plot(loss_train_history, label='train')
        if loss_test_history is not None:
            plt.plot(loss_test_history, label='test')

        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(self.checkpoint_path + "images/" + 'loss.png')
        print("Time to save figures: ", timer.get_time())


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




