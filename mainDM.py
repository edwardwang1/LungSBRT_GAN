# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import math
from inspect import isfunction
from functools import partial
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.animation as animation
from modelsDM3d import *
from PIL import Image
import requests
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

# %matplotlib inline
import matplotlib.pyplot as plt
import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from dataset import VolumesFromList
from torchvision.utils import save_image
import os

def saveImgNby3(arrs, ct, save_path, labels=None):
    aspect = 1.
    n = len(arrs)  # number of rows
    m = 3
    bottom = 0.1
    left = 0.05
    top = 1. - bottom
    right = 1. - 0.18
    fisasp = (1 - bottom - (1 - top)) / float(1 - left - (1 - right))
    # widthspace, relative to subplot size
    wspace = 0  # set to zero for no spacing
    hspace = wspace / float(aspect)
    # fix the figure height
    figheight = 10 / 5  # inch
    figwidth = (m + (m - 1) * wspace) / float((n + (n - 1) * hspace) * aspect) * figheight * fisasp

    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                        wspace=wspace, hspace=hspace)

    short_dim = arrs[0].shape[0]
    long_dim = arrs[0].shape[1]
    diff_dim = int((long_dim - short_dim) / 2)

    for i in range(len(arrs)):
        arrs[i] = arrs[i][:, diff_dim:long_dim - diff_dim, diff_dim:long_dim - diff_dim]

    ct = ct[:, diff_dim:long_dim - diff_dim, diff_dim:long_dim - diff_dim]

    individual_mins = []
    individual_maxs = []

    for i in range(len(arrs)):
        individual_mins.append(np.min(arrs[i]))
        individual_maxs.append(np.max(arrs[i]))

    min_overall = np.min(individual_mins)
    max_overall = np.max(individual_maxs)

    # Get best slice location
    index_of_max = np.where(arrs[0] == np.max(arrs[0]))

    ax_loc = index_of_max[0][0]
    sag_loc = index_of_max[1][0]
    cor_loc = index_of_max[2][0]

    images = []
    for volume in arrs:
        images.append(volume[ax_loc, :, :])
        images.append(np.flipud(volume[:, sag_loc, :]))
        images.append(np.flipud(volume[:, :, cor_loc]))

    images_ct = []
    images_ct.append(ct[ax_loc, :, :])
    images_ct.append(ct[:, sag_loc, :][::-1, ::-1])
    images_ct.append(ct[:, :, cor_loc][::-1, ::-1])

    min_threshold = 0
    for i, ax in enumerate(axes.flatten()):
        transparency_mask = (images[i] > min_threshold).astype(int) * 0.99
        ax.imshow(images[i], cmap="rainbow", vmin=min_overall, vmax=max_overall, alpha=transparency_mask)
        if i % 3 == 0:
            ax.text(2, 5, labels[int(i / 3)], fontsize=6, fontweight="bold")
        ax.imshow(images_ct[i % 3], cmap='gray', alpha=0.7)
        ax.axis('off')

    norm = matplotlib.colors.Normalize(vmin=min_threshold, vmax=max_overall)

    sm = matplotlib.cm.ScalarMappable(cmap="rainbow", norm=norm)

    array_for_colorbar = None
    for i in range(len(arrs)):
        if np.max(arrs[i]) == max_overall:
            array_for_colorbar = arrs[i]

    array_for_colorbar = np.clip(array_for_colorbar, a_min=min_threshold, a_max=None)

    sm.set_array([array_for_colorbar])

    cax = fig.add_axes([right + 0.035, bottom, 0.035, top - bottom])
    fig.colorbar(sm, cax=cax)

    #plt.savefig(save_path, format="png", dpi=100, bbox_inches='tight')

    try:
        plt.savefig(save_path, format="png", dpi=100, bbox_inches='tight')
    except:
        print("Error saving image: " + save_path)

    #plt.close(fig)
    return fig

def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200, 200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

def saveImages(listOfImages, save_path):
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(listOfImages[0][0, 0, 46, :, :])
    axarr[0, 1].imshow(listOfImages[int(0.3333 * len(listOfImages))][0, 0, 46, :, :])
    axarr[1, 0].imshow(listOfImages[int(0.666 * len(listOfImages))][0, 0, 46, :, :])
    axarr[1, 1].imshow(listOfImages[-1][0, 0, 46, :, :])

    plt.savefig(save_path, format="png", dpi=100, bbox_inches='tight')
    plt.close(f)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, condition, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    predicted_noise = denoise_model(x_noisy, condition, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

# define function
# def custom_transforms(examples):
#     examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
#     del examples["image"]
#
#     return examples

@torch.no_grad()
def p_sample(model, x, condition, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, condition, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 but save all images:

@torch.no_grad()
def p_sample_loop(model, condition, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    #for i in tqdm.tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, condition, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, condition, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, condition, shape=(batch_size, channels, image_size, image_size, image_size))

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

if __name__ == '__main__':
    timesteps = 200

    # define beta schedule
    #betas = linear_beta_schedule(timesteps=timesteps)
    betas = cosine_beta_schedule(timesteps=timesteps)


    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


    # use seed for reproducability
    torch.manual_seed(0)

    dim = 8
    channels = 1
    batch_size = 1

    save_and_sample_every = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Unet(
        dim=dim,
        channels=channels,
        dim_mults=(1, 2, 4,),
        out_dim=1
    )
    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=2e-4)

    """Let's start training!"""

    num_epochs = 300

    data_dir = "P:/My Documents/LungTumourRadiotherapy/NumpyFilesV1/LIMBUS_PTV_IGTV56"
    patientList_dir = "P:/My Documents/LungTumourRadiotherapy/LungSBRT_GAN/Data/ListsOfFolds"
    results_folder = "P:/My Documents/LungTumourRadiotherapy/LungSBRT_GAN/DiffusionModelOutput"

    train_dataset = VolumesFromList(data_dir, patientList_dir, valFold=3, testingHoldoutFold=4, test=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    epoch_loop = tqdm.tqdm(range(num_epochs + 1))


    for epoch in epoch_loop:
        for step, volumes in enumerate(train_loader):
            optimizer.zero_grad()

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            real_dose = volumes[:, 0, :, :, :].unsqueeze(1).float().to(device)
            real_dose = (real_dose / 400) - 1

            est_dose = volumes[:, 1, :, :, :].unsqueeze(1).float().to(device)
            oars = volumes[:, 2, :, :, :].unsqueeze(1).float().to(device)
            ct = volumes[:, 3, :, :, :].unsqueeze(1).float().to(device)

            condition = torch.cat([est_dose, ct, oars], dim=1)

            #p_losses(denoise_model, x_start, condition, t, noise=None, loss_type="l1"):
            loss = p_losses(model, real_dose, None, t, loss_type="l2")

            if step % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            optimizer.step()

            #save generated images
            #if step != 0 and step % save_and_sample_every == 0:
            # print("saving images")
            if step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                #batches = num_to_groups(4, batch_size)
                all_images_list = sample(model, None, image_size=92, batch_size=1, channels=1)

                #print(all_images_list[0].shape)
                try:
                    saveImages(all_images_list, os.path.join(results_folder, f'sample-{epoch}_{milestone}.png'))
                except:
                    print("error saving images")


                # #print(len(img))
                # all_images_list = list(map(lambda n: sample(model, condition, image_size=92, batch_size=batch_size, channels=1), batches))
                # all_images = torch.cat(all_images_list, dim=0)
                # all_images = (all_images + 1) * 0.5
                # #print(all_images)
                # # plot = saveImgNby3(
                # #     img[0, 0, :, :, :], real_dose[0, 0, :, :, :], oars[0, 0, :, :, :], ct[0, 0, :, :, :],
                # #         os.path.join(results_folder, f'sample-{milestone}.png'),)
                # save_image(all_images, os.path.join(results_folder, f'sample-{milestone}.png'), nrow=6)
    #
    #
    #
    # # sample 64 images
    # samples = sample(model, image_size=image_size, batch_size=64, channels=channels)
    #
    # # show a random one
    # random_index = 5
    # plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
    #
    #
    #
    # random_index = 53
    #
    # fig = plt.figure()
    # ims = []
    # for i in range(timesteps):
    #     im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
    #     ims.append([im])
    #
    # animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    # animate.save('diffusion.gif')
    # plt.show()
