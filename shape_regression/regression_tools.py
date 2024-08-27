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
from matplotlib import gridspec
from scipy import stats
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix

class RegressionTools:
    def __init__(self, args, label_names, train_loader, test_loader, model, opt, scheduler=None, clip_parameters=False,
                 include_faces=False):
        torch.cuda.empty_cache()
        self.device = torch.device("cuda")

        self.dataset_name = args["dataset_name"]
        self.model_name = type(model).__name__

        self.do_classification = False
        if args["use_category_thresholds"] is not None:
            self.do_classification = True

        if "data_parallel" in args and not args["data_parallel"]:
            self.model = model.to(self.device)
        else:
            self.model = nn.DataParallel(model.to(self.device))
        # self.model = model.to(self.device)

        self.label_names = label_names

        self.num_outputs = len(label_names)
        self.test_loader = test_loader
        self.train_loader = train_loader

        if self.do_classification:
            self.num_classes = len(args["use_category_thresholds"]) + 1
            # create weights here
            weights = None
            if self.train_loader.dataset.imbalanced_weighting is not None:
                imbalanced_weighting = self.train_loader.dataset.imbalanced_weighting
                weights = imbalanced_weighting.get_weights(np.arange(0, self.num_classes, step=1))
            self.loss_criterion = torch.nn.CrossEntropyLoss(torch.Tensor(weights).to(self.device))
        else:
            self.loss_criterion = torch.nn.MSELoss(reduction='mean')


        self.clip_parameters = clip_parameters
        self.clip_threshold = 1.0

        self.include_faces = include_faces

        cudnn.benchmark = True

        self.opt = opt
        self.scheduler = scheduler

        self.args = args
        self.checkpoint_path = ("checkpoints/" + self.dataset_name + "/" + self.model_name + "/" +
                                util.get_date_name() + "_" + args['exp_name'] + "_" + args['notes'] + "/")

        print("Saving checkpoints to: ", self.checkpoint_path)
        self.io = self.create_checkpoints(args)
        self.gradient_accumulation_steps = args["grad_acc_steps"]
        self.NUM_PLOT_POINTS = 7500
        self.default_alpha = 1000.0 / self.NUM_PLOT_POINTS

        # self.io.cprint("Total training data: " + str(self.train_loader.))
        print("Parameters in model: ", get_num_parameters(self.model))

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
        for batch_idx, (data, label) in enumerate(tqdm(data_loader)):
            # calculate weights if appropriate
            weight = 1
            if data_loader.dataset.imbalanced_weighting is not None:
                weight = data_loader.dataset.imbalanced_weighting.get_weights(label.numpy())
                weight = torch.Tensor(weight)
                weight = weight.to(self.device)

            data, label = data.to(self.device), label.to(self.device)

            preds = self.model(data)

            if self.do_classification:
                loss_labels = label.long()
                loss_preds = preds.clone()
                if len(label.shape) == 3:
                    loss_labels = loss_labels.reshape((label.shape[0] * label.shape[1]))
                    loss_preds = loss_preds.reshape((preds.shape[0] * preds.shape[1], preds.shape[2]))
                # Note: weight is already incorporated into the loss for classification
            else:
                loss_labels = weight * label
                loss_preds = weight * preds

            if training:
                loss = self.loss_criterion(loss_preds, loss_labels) / self.gradient_accumulation_steps
                loss.backward()

                if batch_idx % self.gradient_accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                    if self.clip_parameters:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_threshold)
                    self.opt.step()
                    self.opt.zero_grad()
            else:
                loss = self.loss_criterion(loss_preds, loss_labels)

            batch_size = data.size()[0]
            count += batch_size
            train_loss += loss.item() * batch_size

            label = label.cpu().numpy()
            preds = preds.detach().cpu().numpy()
            # for vertices, outputs as batch x dim x points. swap to batch*points x dim
            if len(label.shape) == 3:
                label = label.reshape((label.shape[0] * label.shape[1], label.shape[2]), order="F")
                preds = preds.reshape((preds.shape[0] * preds.shape[1], preds.shape[2]), order="F")

            # convert for classification
            if self.do_classification:
                preds = np.argmax(preds, axis=-1)
                preds = preds[..., np.newaxis]

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
            weight = 1.0
            if data_loader.dataset.imbalanced_weighting is not None:
                weight = data_loader.dataset.imbalanced_weighting.get_weights(label.numpy())
                weight = torch.Tensor(weight)
                weight = weight.to(self.device)

            vertices, faces, label = vertices.to(self.device), faces.to(self.device), label.to(self.device)

            try:
                preds = self.model(vertices, faces)
            except Exception as e:
                print("Training: Error calculating model. Skipping")
                print(e)
                continue

            # if self.do_classification:
            #     preds = torch.argmax(preds)

            if training:
                if len(label.shape) == 3:
                    # TODO note this assumes batch size 1
                    loss = self.loss_criterion(weight * preds,
                                               weight * label) / self.gradient_accumulation_steps
                    # loss_indices = util.get_permutation_for_list(preds[0], 4096)
                    # loss = self.loss_criterion(weight * preds[:, loss_indices, :],
                    #                            weight * label[:, loss_indices, :]) / self.gradient_accumulation_steps

                else:
                    loss = self.loss_criterion(weight * preds, weight * label) / self.gradient_accumulation_steps
                loss.backward()

                if batch_idx % self.gradient_accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                    # print(batch_idx)
                    if self.clip_parameters:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_threshold)
                    self.opt.step()
                    self.opt.zero_grad()
            else:
                # TODO this assumes all vertex lengths are equal or batch size 1
                loss = self.loss_criterion(weight * preds, weight * label)

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
                # TODO These all have different shapes!!
                # TODO below assumes batch size 1
                # r2_indices = np.arange(start=0, stop=len(preds), step=int(np.ceil(len(preds) / R2_SAMPLES)), dtype=np.int64)
                # label = label[r2_indices]
                # preds = preds[r2_indices]

            train_true.append(label)
            train_pred.append(preds)

        return train_loss, count, train_pred, train_true

    def train(self, args, do_test=True, plot_every_n_epoch=-1):
        best_res_mag = 0
        loss_train_history = []
        res_train_history = []
        acc_train_history = []
        loss_test_history = []
        res_test_history = []
        acc_test_history = []
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
            train_pred = np.concatenate(train_pred)
            train_pred = np.nan_to_num(train_pred) # TODO this may not be the best option

            if self.scheduler is not None:
                if self.args["scheduler"] == "plateau":
                    self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()

            res_train = metrics.r2_score(y_true=train_true, y_pred=train_pred, multioutput='raw_values') #num_val x dims
            acc = get_accuracy_tolerance(preds=train_pred, actual=train_true, tolerance=0.05)

            outstr = '========\nTrain Epoch:' + str(epoch)
            outstr += '\n\tLoss: ' + str(test_loss * 1.0 / count)
            outstr += '\n\tr2: ' + str(res_train)
            outstr += '\n\tacc: ' + str(acc)
            outstr += '\n\tlr: ' + str(self.opt.param_groups[0]['lr'])
            # outstr = ('========\nTrain Epoch: %d '
            #           '\n\tLoss: %.6f '
            #           '\n\tr2: %s '
            #           '\n\tlr: %s '
            #           '\n\tacc: %s' %
            #           (epoch,
            #            train_loss * 1.0 / count,
            #            str(res_train),
            #            self.opt.param_groups[0]['lr'],
            #            str(acc)))
            self.io.cprint(outstr)

            loss_train_history.append(train_loss * 1.0 / count)
            res_train_history.append(res_train)
            acc_train_history.append(acc)

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
                test_pred = np.nan_to_num(test_pred)  # TODO this may not be the best option


                res_test = metrics.r2_score(y_true=test_true, y_pred=test_pred, multioutput='raw_values')
                acc_test = get_accuracy_tolerance(preds=test_pred, actual=test_true, tolerance=0.05)

                outstr = '--\n\tTest:'
                outstr += '\n\t\tLoss: ' + str(test_loss * 1.0 / count)
                outstr += '\n\t\tr2: ' + str(res_test)
                outstr += '\n\t\tacc: ' + str(acc_test)
                self.io.cprint(outstr)

                # ('--\n\tTest: \n\t\tLoss: %.6f \n\t\tr2: %s\n=========' %
                #  (test_loss * 1.0 / count, str(res_test)))
                loss_test_history.append(train_loss * 1.0 / count)
                res_test_history.append(res_test)
                acc_test_history.append(acc_test)

            # Logging
            if (plot_every_n_epoch >= 1 and epoch % plot_every_n_epoch == 0) or epoch + 1 == args['epochs']:
                timer = Stopwatch()
                timer.start()
                self.save_r2(phase='train', epoch=epoch, r2_list=res_train, true_list=train_true, pred_list=train_pred, labels=labels)

                if do_test:
                    self.save_r2(phase='test', epoch=epoch, r2_list=res_test, true_list=test_true, pred_list=test_pred, labels=labels)

                print("Time to save figures: ", timer.get_time())

            if do_test:
                if self.do_classification:
                    self.save_history_per_label(labels=labels, name="Accuracy",
                                                train_history=acc_train_history, test_history=acc_test_history)
                else:
                    self.save_history_per_label(labels=labels, name="R2",
                                                train_history=res_train_history, test_history=res_test_history)
                self.save_history(name="Loss",
                                    train_history=loss_train_history, test_history=loss_test_history)
            else:
                if self.do_classification:
                    self.save_history_per_label(labels=labels, name="Accuracy", train_history=acc_train_history)
                else:
                    self.save_history_per_label(labels=labels, name="R2", train_history=res_train_history)
                self.save_history(name="Loss", train_history=loss_train_history)

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


    def save_r2(self, phase, epoch, r2_list, true_list, pred_list, labels):
        # fig, axes = plt.subplots(len(r2_list), 1)
        # if len(r2_list) == 1:
        #     axes = [axes]
        # for i in range(len(r2_list)):
        #     ax = axes[i]
        #     self.plot_r2_on_ax(ax, phase=phase, true=true_list[:, i], pred = pred_list[:, i], label=labels[i])
        #
        # plt.savefig(self.checkpoint_path + "images/" + phase + '_confusion_' + str(epoch) + '.png')

        for i in range(len(r2_list)):
            true = true_list[:, i]
            pred = pred_list[:, i]
            plot_indices = np.arange(start=0, stop=len(true), step=int(np.ceil(len(true) / self.NUM_PLOT_POINTS)),
                                     dtype=np.int64)
            fig = self.create_r2_kde_fig(phase=phase, true=true[plot_indices], pred=pred[plot_indices], label=labels[i])
            fig.savefig(self.checkpoint_path + "images/" + phase + '_confusion_' + labels[i] + "_" + str(epoch) + '.png')

    def plot_r2_on_ax(self, ax, phase, true, pred, label):
        error = pred - true
        # plot_indices = np.arange(start=0, stop=len(error), step=int(np.ceil(len(error) / self.NUM_PLOT_POINTS)),
        #                          dtype=np.int64)
        ax.scatter(true, error, alpha=self.default_alpha)
        ax.hlines(y=0, xmin=min(true), xmax=max(true), color="red")
        ax.set_xlabel(phase + " True", fontsize=18)
        ax.set_ylabel(label + " Error", fontsize=18)

    def plot_confusion_matrix_on_ax(self, ax, true, pred, phase, label):
        # Build confusion matrix
        classes = np.arange(0, self.num_classes, step=1)
        cf_matrix = confusion_matrix(true, pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                             columns=[i for i in classes])
        sn.heatmap(df_cm, annot=True, ax=ax)
        ax.set_xlabel(phase, fontsize=18)
        ax.set_ylabel(label, fontsize=18)

    def create_r2_kde_fig(self, phase, true, pred, label, ratio=5):
        # TODO need to sample from pred and true
        ##### Instantiate grid ########################
        # Set up 4 subplots and aspect ratios as axis objects using GridSpec:
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, ratio], height_ratios=[ratio, 1])
        # Add space between scatter plot and KDE plots to accommodate axis labels:
        gs.update(hspace=0.3, wspace=0.3)

        fig = plt.figure(figsize=[12, 10])  # Set background canvas colour to White instead of grey default
        fig.patch.set_facecolor('white')

        error = pred - true

        xmin = np.min(true)
        xmax = np.max(true)
        ymin = np.min(error)
        ymax = np.max(error)

        ax = plt.subplot(gs[0, 1])  # Instantiate scatter plot area and axis range
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.yaxis.labelpad = 10  # adjust space between x and y axes and their labels if needed

        axl = plt.subplot(gs[0, 0], sharey=ax)  # Instantiate left KDE plot area
        axl.get_xaxis().set_visible(False)  # Hide tick marks and spines
        axl.get_yaxis().set_visible(False)
        axl.spines["right"].set_visible(False)
        axl.spines["top"].set_visible(False)
        axl.spines["bottom"].set_visible(False)

        axb = plt.subplot(gs[1, 1], sharex=ax)  # Instantiate bottom KDE plot area
        axb.get_xaxis().set_visible(False)  # Hide tick marks and spines
        axb.get_yaxis().set_visible(False)
        axb.spines["right"].set_visible(False)
        axb.spines["top"].set_visible(False)
        axb.spines["left"].set_visible(False)

        axc = plt.subplot(gs[1, 0])  # Instantiate legend plot area
        axc.axis('off')  # Hide tick marks and spines

        ##### Actually Plot ############################
        if self.do_classification:
            self.plot_confusion_matrix_on_ax(ax=ax, phase=phase, label=label, true=true, pred=pred)
        else:
            self.plot_r2_on_ax(ax, phase, true, pred, label)

        kde = stats.gaussian_kde(true)
        xx = np.linspace(xmin, xmax, 1000)
        kde_val = kde(xx)
        axb.plot(xx, kde_val)
        axb.fill_between(xx, kde_val, alpha=0.3)
        axb.set_ylabel("KDE") # This isn't showing for some reason
        axb.set_ylim(0, np.max(kde_val) * 1.1)

        # also plot weights if appropriate
        if phase == "train":
            data_loader = self.train_loader
        else:
            data_loader = self.test_loader
        if data_loader.dataset.has_weights:
            axb2 = axb.twinx()
            axb2.set_ylabel("Weights")
            imbalanced_weighting = data_loader.dataset.imbalanced_weighting
            weights = imbalanced_weighting.get_weights(xx)
            axb2.set_ylim(0, np.max(weights))
            axb2.plot(xx, weights, c='orange')
            axb2.fill_between(xx, weights, color='orange', alpha=0.3)


        kde = stats.gaussian_kde(error)
        yy = np.linspace(ymin, ymax, 1000)
        axl.plot(kde(yy), yy)
        axl.fill_between(kde(yy), yy, alpha=0.3)

        return fig

    def save_history_per_label(self, name, labels, train_history, test_history=None):
        fig, axes = plt.subplots(len(labels), 1)
        if len(labels) == 1:
            axes = [axes]
        fig.set_size_inches(8, 6)

        for i in range(len(labels)):
            ax = axes[i]
            # ax.subplot(len(labels), 1, i + 1)
            train_history = np.array(train_history)
            ax.plot(train_history[:, i], label='train')

            if test_history is not None:
                test_history = np.array(test_history)
                plt.plot(test_history[:, i], label='test')

            ax.set_ylabel(labels[i] + " " + name)
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(visible=True, which='major', axis='y')

        fig.savefig(self.checkpoint_path + "images/" + name + '.png')

    def save_history(self, name, train_history, test_history=None):
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)

        ax.plot(train_history, label='train')

        ax.plot(test_history, label='test')

        ax.set_ylabel(name)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(visible=True, which='major', axis='y')

        fig.savefig(self.checkpoint_path + "images/" + name + '.png')


    def save_loss(self, labels, res_train_history, loss_train_history, res_test_history=None, loss_test_history=None):
        timer = Stopwatch()
        timer.start()

        fig, axes = plt.subplots(len(labels), 1)
        if len(labels) == 1:
            axes = [axes]
        fig.set_size_inches(8, 6)

        for i in range(len(labels)):
            ax = axes[i]
            # ax.subplot(len(labels), 1, i + 1)
            train_res_history = np.array(res_train_history)
            ax.plot(train_res_history[:, i], label='train')

            if res_test_history is not None:
                test_res_history = np.array(res_test_history)
                plt.plot(test_res_history[:, i], label='test')

            ax.set_ylabel(labels[i] + " Train R2")
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(visible=True, which='major', axis='y')

        fig.savefig(self.checkpoint_path + "images/" + 'r2_train.png')

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)
        ax.plot(loss_train_history, label='train')
        if loss_test_history is not None:
            ax.plot(loss_test_history, label='test')

        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        ax.legend()
        fig.savefig(self.checkpoint_path + "images/" + 'loss.png')
        # print("Time to save figures: ", timer.get_time())


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


def get_num_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params

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


