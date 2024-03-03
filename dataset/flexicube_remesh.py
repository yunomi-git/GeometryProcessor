# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import argparse
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
import os
from flexicubes.examples.util import *
import flexicubes.examples.render as render
import flexicubes.examples.loss as loss
import imageio

import sys

sys.path.append('..')
from flexicubes.flexicubes import FlexiCubes
import paths

###############################################################################
# Functions adapted from https://github.com/NVlabs/nvdiffrec
###############################################################################

def lr_schedule(iter):
    return max(0.0, 10 ** (-(iter) * 0.0002))  # Exponential falloff from [1.0, 0.1] over 5k epochs.

out_dir = "./cube_out/"
ref_mesh = paths.get_thingiverse_stl_path(6, get_by_order=True)
iter = 600
batch = 8
train_res = [2048, 2048]
learning_rate = 0.01
voxel_grid_res = 128
sdf_loss = True
develop_reg = True
sdf_regularizer = 0.3
display_res = [512, 512]
save_interval = 20

if __name__ == "__main__":
    device = 'cuda'

    os.makedirs(out_dir, exist_ok=True)
    glctx = dr.RasterizeGLContext()

    # Load GT mesh
    gt_mesh = load_mesh(ref_mesh, device)
    gt_mesh.auto_normals()  # compute face normals for visualization

    # ==============================================================================================
    #  Create and initialize FlexiCubes
    # ==============================================================================================
    fc = FlexiCubes(device)
    x_nx3, cube_fx8 = fc.construct_voxel_grid(voxel_grid_res)
    x_nx3 *= 2  # scale up the grid so that it's larger than the target object

    sdf = torch.rand_like(x_nx3[:, 0]) - 0.1  # randomly init SDF
    sdf = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
    # set per-cube learnable weights to zeros
    weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device='cuda')
    weight = torch.nn.Parameter(weight.clone().detach(), requires_grad=True)
    deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)

    #  Retrieve all the edges of the voxel grid; these edges will be utilized to
    #  compute the regularization loss in subsequent steps of the process.
    all_edges = cube_fx8[:, fc.cube_edges].reshape(-1, 2)
    grid_edges = torch.unique(all_edges, dim=0)

    # ==============================================================================================
    #  Setup optimizer
    # ==============================================================================================
    optimizer = torch.optim.Adam([sdf, weight, deform], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x))

    # ==============================================================================================
    #  Train loop
    # ==============================================================================================
    for it in range(iter):
        optimizer.zero_grad()
        # sample random camera poses
        mv, mvp = render.get_random_camera_batch(batch, iter_res=train_res, device=device, use_kaolin=False)
        # render gt mesh
        target = render.render_mesh_paper(gt_mesh, mv, mvp, train_res)
        # extract and render FlexiCubes mesh
        grid_verts = x_nx3 + (2 - 1e-8) / (voxel_grid_res * 2) * torch.tanh(deform)
        vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, voxel_grid_res, beta_fx12=weight[:, :12],
                                    alpha_fx8=weight[:, 12:20],
                                    gamma_f=weight[:, 20], training=True)
        flexicubes_mesh = Mesh(vertices, faces)
        buffers = render.render_mesh_paper(flexicubes_mesh, mv, mvp, train_res)

        # evaluate reconstruction loss
        mask_loss = (buffers['mask'] - target['mask']).abs().mean()
        depth_loss = ((
                    (((buffers['depth'] - (target['depth'])) * target['mask']) ** 2).sum(-1) + 1e-8)).sqrt().mean() * 10

        t_iter = it / iter
        sdf_weight = sdf_regularizer - (sdf_regularizer - sdf_regularizer / 20) * min(1.0, 4.0 * t_iter)
        reg_loss = loss.sdf_reg_loss(sdf, grid_edges).mean() * sdf_weight  # Loss to eliminate internal floaters that are not visible
        reg_loss += L_dev.mean() * 0.7
        reg_loss += (weight[:, :20]).abs().mean() * 0.1
        total_loss = mask_loss + depth_loss + reg_loss

        if sdf_loss:  # optionally add SDF loss to eliminate internal structures
            with torch.no_grad():
                pts = sample_random_points(1000, gt_mesh)
                gt_sdf = compute_sdf(pts, gt_mesh.vertices, gt_mesh.faces)
            pred_sdf = compute_sdf(pts, flexicubes_mesh.vertices, flexicubes_mesh.faces)
            total_loss += torch.nn.functional.mse_loss(pred_sdf, gt_sdf) * 2e3

        # optionally add developability regularizer, as described in paper section 5.2
        if develop_reg:
            reg_weight = max(0, t_iter - 0.8) * 5
            if reg_weight > 0:  # only applied after shape converges
                reg_loss = loss.mesh_developable_reg(flexicubes_mesh).mean() * 10
                reg_loss += (deform).abs().mean()
                reg_loss += (weight[:, :20]).abs().mean()
                total_loss = mask_loss + depth_loss + reg_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if (it % save_interval == 0 or it == (iter - 1)):  # save normal image for visualization
            with torch.no_grad():
                # extract mesh with training=False
                vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, voxel_grid_res, beta_fx12=weight[:, :12],
                                            alpha_fx8=weight[:, 12:20],
                                            gamma_f=weight[:, 20], training=False)
                flexicubes_mesh = Mesh(vertices, faces)

                flexicubes_mesh.auto_normals()  # compute face normals for visualization
                mv, mvp = render.get_rotate_camera(it // save_interval, iter_res=display_res, device=device,
                                                   use_kaolin=False)
                val_buffers = render.render_mesh_paper(flexicubes_mesh, mv.unsqueeze(0), mvp.unsqueeze(0),
                                                       display_res, return_types=["normal"], white_bg=True)
                val_image = ((val_buffers["normal"][0].detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)

                gt_buffers = render.render_mesh_paper(gt_mesh, mv.unsqueeze(0), mvp.unsqueeze(0), display_res,
                                                      return_types=["normal"], white_bg=True)
                gt_image = ((gt_buffers["normal"][0].detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(out_dir, '{:04d}.png'.format(it)),
                                np.concatenate([val_image, gt_image], 1))
                print(f"Optimization Step [{it}/{iter}], Loss: {total_loss.item():.4f}")

    # ==============================================================================================
    #  Save ouput
    # ==============================================================================================
    mesh_np = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(),
                              process=False)
    mesh_np.export(os.path.join(out_dir, 'output_mesh.stl'))