import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import diffusion_net
import paths
from heuristic_prediction.mesh_dataloader import DiffusionNetDataset

# parser.add_argument("--split_size", type=int, help="how large of a training set per-class default: 10", default=10)
split_size = 10

# system things
device = torch.device('cuda:0')
dtype = torch.float32

# model
input_features = 'hks'  # one of ['xyz', 'hks']
k_eig = 64

# training settings
n_epoch = 200
lr = 1e-3
num_decays = 8
decay_every = int(n_epoch / num_decays) #50
decay_rate = 0.5
augment_random_rotate = (input_features == 'xyz')
# label_smoothing_fac = 0.2

# Important paths
base_path = os.path.dirname(__file__)
# op_cache_dir = os.path.join(base_path, "data", "op_cache")


# dataset_path = paths.TRAINING_DATA_PATH
dataset_path = paths.DATA_PATH + "data_th5k_norm/"
op_cache_dir = dataset_path + "op_cache"
# === Load datasets

# def filter_criteria(mesh, instance_data) -> bool:
#     # if mesh_data["vertices"] > 1e4:
#     #     return true
#     # if instance_data["vertices"] < 1e3:
#     #     return False
#     # if instance_data["vertices"] > 1e5:
#     #     return False
#     if math.isnan(instance_data["thickness_violation"]):
#         return False
#     # if instance_data["scale"] > 1000:
#     #     return False
#
#     return True

# Train dataset
train_dataset = DiffusionNetDataset(dataset_path, split_size=split_size,
                                    k_eig=k_eig, op_cache_dir=op_cache_dir)
train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

# Test dataset
test_dataset = DiffusionNetDataset(dataset_path, split_size=None,
                                   k_eig=k_eig, op_cache_dir=op_cache_dir,
                                   exclude_dict=train_dataset.entries)
test_loader = DataLoader(test_dataset, batch_size=None)

# === Create the model

C_in = {'xyz': 3, 'hks': 16}[input_features]  # dimension of input features
NUM_OUTPUTS = 1

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=NUM_OUTPUTS,
                                          C_width=128,
                                          N_block=8,
                                          last_activation=None, #lambda x : torch.nn.functional.tanh(x),
                                          outputs_at='global_mean',
                                          # diffusion_method='implicit_dense',
                                          dropout=False)

model = model.to(device)

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def get_targets(data):
    verts, faces, frames, mass, L, evals, evecs, gradX, gradY, output = data
    targets_vector = np.array([
        # output["overhang_violation"],
        # output["stairstep_violation"],
        output["thickness_violation"],
        # output["gap_violation"],
    ])
    targets_vector = torch.from_numpy(targets_vector)
    targets = targets_vector.to(device).float()
    return targets

def get_preds(data, model, is_training):
    # Get data
    verts, faces, frames, mass, L, evals, evecs, gradX, gradY, output = data

    # Move to device
    verts = verts.to(device)
    faces = faces.to(device)
    # frames = frames.to(device)
    mass = mass.to(device)
    L = L.to(device)
    evals = evals.to(device)
    evecs = evecs.to(device)
    gradX = gradX.to(device)
    gradY = gradY.to(device)

    # Randomly rotate positions
    if augment_random_rotate and is_training:
        verts = diffusion_net.utils.random_rotate_points(verts)

    # Construct features
    if input_features == 'xyz':
        features = verts
    else:  # input_features == 'hks':
        features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16) # TODO autoscale here

    # Apply the model
    preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
    return preds

def train_epoch(epoch):
    loss_func = torch.nn.MSELoss()
    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

            # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()

    total_percent_correct = 0
    avg_loss = 0
    total_num = 0
    for data in tqdm(train_loader):
        preds = get_preds(data, model, is_training=True)
        targets = get_targets(data)
        if targets.isnan().any():
            # print("NAN Detected. Skipping")
            # print(data[9]["mesh_relative_path"])
            continue
        # Evaluate loss
        loss = loss_func(preds, targets)
        loss.backward()

        # track accuracy - count how many are within error
        correct_prediction_vector = torch.isclose(targets, preds, atol=1e-3)
        percent_correct = torch.sum(correct_prediction_vector) / (NUM_OUTPUTS * 1.0)
        total_percent_correct += percent_correct
        total_num += 1
        avg_loss += loss

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_acc = total_percent_correct / total_num
    avg_loss /= total_num
    return train_acc, avg_loss


# Do an evaluation pass on the test dataset
def do_test():
    model.eval()

    total_percent_correct = 0
    total_num = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            preds = get_preds(data, model, is_training=False)
            targets = get_targets(data)
            if targets.isnan().any():
                print("NAN Detected. Skipping")
                print(data["mesh_relative_path"])
                continue
            # track accuracy
            correct_prediction_vector = torch.isclose(targets, preds, atol=1e-4)
            percent_correct = torch.sum(correct_prediction_vector) / (NUM_OUTPUTS * 1.0)
            total_percent_correct += percent_correct
            total_num += 1

    test_acc = total_percent_correct / total_num
    return test_acc

print("Training...")

for epoch in range(n_epoch):
    train_acc, avg_loss = train_epoch(epoch)
    print("")
    print("Train Loss: ", avg_loss.item())
    # test_acc = do_test()
    test_acc = 0
    print("")
    print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100 * train_acc, 100 * test_acc))

# Test
test_acc = do_test()
print("Overall test accuracy: {:06.3f}%".format(100 * test_acc))
