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
from util import IOStream



class RegressionTools:
    def __init__(self, args, label_names, train_loader, test_loader, model, opt, scheduler=None, clip_parameters=False):
        self.device = torch.device("cuda")
        self.seed_all(args["seed"])

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
        self.checkpoint_path = ("checkpoints/" + self.model_name + "/" +
                                util.get_date_name() + "_" + args['exp_name'] + "/")
        print("Saving checkpoints to: ", self.checkpoint_path)
        self.io = self.create_checkpoints(args)
        self.gradient_accumulation_steps = args["grad_acc_steps"]


    def create_checkpoints(self, args):
        Path(self.checkpoint_path + 'models/').mkdir(parents=True, exist_ok=True)
        # save the arguments
        with open(self.checkpoint_path + "args.json", "w") as f:
            json.dump(args, f, indent=4)

        io = IOStream(self.checkpoint_path + 'run.log')
        return io

    def seed_all(self, seed):
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

    def train(self, args, do_test=True):
        best_res_mag = 0
        for epoch in range(args['epochs']):
            ####################
            # Train
            ####################
            train_loss = 0.0
            count = 0.0
            self.model.train()
            train_pred = []
            train_true = []
            with torch.enable_grad():
                for batch_idx, (data, label) in enumerate(tqdm(self.train_loader)):
                    data, label = data.to(self.device), label.to(self.device)
                    data = data.permute(0, 2, 1) # so, the input data shape is [batch, 3, 1024]

                    preds = self.model(data)
                    loss = self.loss_criterion(preds, label) / self.gradient_accumulation_steps
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
                self.scheduler.step()

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)

            res = metrics.r2_score(y_true=train_true, y_pred=train_pred, multioutput='raw_values')
            acc = get_accuracy_tolerance(preds=train_pred, actual=train_true, tolerance=0.1)

            outstr = ('========\nTrain Epoch: %d '
                      '\n\tLoss: %.6f '
                      '\n\tr2: %s '
                      '\n\tlr: %s '
                      '\n\tacc: %s' %
                      (epoch, train_loss * 1.0 / count, str(res), self.opt.param_groups[0]['lr'], str(acc)))
            self.io.cprint(outstr)

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
                    for data, label in tqdm(self.test_loader):
                        data, label = data.to(self.device), label.to(self.device)
                        data = data.permute(0, 2, 1)
                        batch_size = data.size()[0]
                        preds = self.model(data)
                        loss = self.loss_criterion(preds, label)
                        count += batch_size
                        test_loss += loss.item() * batch_size
                        test_true.append(label.cpu().numpy())
                        test_pred.append(preds.detach().cpu().numpy())

                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
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
        out += name[:3] + "-"
    return out
