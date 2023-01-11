import numpy as np
import os
import tqdm
import argparse, sys

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Custom
from models import Generator, Discriminator, AttentionGenerator
from loss import GDL
from dataset import Volumes, VolumesFromList
from config import load_config

import matplotlib
import matplotlib.pyplot as plt

def getV_XfromTensor(doseVolume, oarVolume, oarCode, doseTarget):
    # doseVolume is a [N, 1, 96, 144, 144] tensor
    # convert doseVolume to a numpy array, remove the channel dimension
    doseVolume = doseVolume.squeeze(1).cpu().numpy()
    # convert oarVolume to a numpy array, remove the channel dimension
    oarVolume = oarVolume.squeeze(1).cpu().numpy()
    oar = oarVolume == oarCode
    greaterThanDose = doseVolume > doseTarget
    V_Xs = []
    for i in range(doseVolume.shape[0]):
        V_Xs.append((greaterThanDose[i] * oar[i]).sum()/oar[i].sum() * 100)
    return np.array(V_Xs)

def saveImg4by3(v1, v2, v3, v4, save_path):
    f, axarr = plt.subplots(4, 3)
    axarr[0, 0].imshow(v1[46, :, :])
    axarr[0, 1].imshow(v1[:, 72, :])
    axarr[0, 2].imshow(v1[:, :, 72])

    axarr[1, 0].imshow(v2[46, :, :])
    axarr[1, 1].imshow(v2[:, 72, :])
    axarr[1, 2].imshow(v2[:, :, 72])

    axarr[2, 0].imshow(v3[46, :, :])
    axarr[2, 1].imshow(v3[:, 72, :])
    axarr[2, 2].imshow(v3[:, :, 72])

    axarr[3, 0].imshow(v4[46, :, :])
    axarr[3, 1].imshow(v4[:, 72, :])
    axarr[3, 2].imshow(v4[:, :, 72])
    plt.savefig(save_path)
    plt.close(f)

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
    figheight = 10  # inch
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
            ax.text(2, 5, labels[int(i / 3)], fontsize=12, fontweight="bold")
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

    plt.savefig(save_path, format="png", dpi=1000, bbox_inches='tight')
    plt.close(fig)

def getDLoss(g, d, real_dose, oars, alt_condition, disc_alt_condition, adv_criterion):
    D_real = d(real_dose, disc_alt_condition, oars)
    #print("D_real: ", torch.mean(D_real))
    D_real_loss = adv_criterion(D_real, torch.ones_like(D_real))
    with torch.no_grad():
        y_fake = g(alt_condition, oars)
        D_fake = d(y_fake.detach(), disc_alt_condition, oars)

    #print("D_fake: ", torch.mean(D_fake))
    D_fake_loss = adv_criterion(D_fake, torch.zeros_like(D_fake))
    D_loss = (D_real_loss + D_fake_loss) / 2
    return D_loss, D_real_loss, D_fake_loss

def getGLoss(g, d, real_dose, oars, alt_condition, disc_alt_condition, adv_criterion, voxel_criterion, alpha, beta):
    y_fake = g(alt_condition, oars)
    D_fake = d(y_fake, disc_alt_condition, oars)
    G_Dcomp_loss_train = adv_criterion(D_fake, torch.ones_like(D_fake))
    G_voxel_loss = voxel_criterion(y_fake, real_dose) / torch.numel(y_fake)
    G_masked_G_voxel_loss = voxel_criterion(y_fake * (oars > 0), real_dose * (oars > 0)) / torch.sum(oars > 0)
    G_loss = G_Dcomp_loss_train + alpha * ((1 - beta) * G_voxel_loss + beta * G_masked_G_voxel_loss)
    return G_loss, G_Dcomp_loss_train, G_voxel_loss, G_masked_G_voxel_loss, y_fake

def train(data_dir, patientList_dir, save_dir, exp_name_base, exp_name, params):
    num_epochs = params["num_epochs"]
    alpha = params["alpha"]
    beta = params["beta"]
    log_interval = params["log_interval"]
    loss_type = params["loss_type"]
    d_update_ratio = params["d_update_ratio"]
    batch_size = params["batch_size"]
    generator_attention = params["generator_attention"]
    alt_condition_volume = params["alt_condition_volume"]
    g_lr = params["g_lr"]
    d_lr = params["d_lr"]
    adv_loss_type = params["adv_loss_type"]
    pretrain_disc = params["pretrain_disc"]
    pretrain_disc_epoch = params["pretrain_disc_epoch"]

    #g_lr = 2e-4
    #d_lr = 2e-4

    oarCodes = {"lung": 1,
                # "heart":2,
                # "eso":3,
                }

    if generator_attention:
        if alt_condition_volume == "ED":
            g = AttentionGenerator(2, 1)
        else:
            g = AttentionGenerator(3, 1)
    else:
        if alt_condition_volume == "ED":
            g = Generator(2, 1)
        else:
            g = Generator(3, 1)

    g.cuda()
    d = Discriminator()
    d.cuda()

    opt_g = optim.Adam(g.parameters(), lr=g_lr, betas=(0.5, 0.999), )
    opt_d = optim.Adam(d.parameters(), lr=d_lr, betas=(0.5, 0.999), )
    if adv_loss_type == "ls":
        adv_criterion = nn.MSELoss(reduction="mean").cuda()
    elif adv_loss_type == "bce":
        adv_criterion = nn.BCEWithLogitsLoss().cuda()
    #
    L2_LOSS_sum = nn.MSELoss(reduction="sum").cuda()

    if loss_type == "l1":
        voxel_criterion = nn.L1Loss(reduction="sum").cuda()
    elif loss_type == "gdl":
        voxel_criterion = GDL()
    elif loss_type == "l2":
        voxel_criterion = nn.MSELoss(reduction="sum").cuda()

    # train_dataset = Volumes(train_dir)
    train_dataset = VolumesFromList(data_dir, patientList_dir, valFold=3, testingHoldoutFold=4, test=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # test_dataset = Volumes(test_dir)
    test_dataset = VolumesFromList(data_dir, patientList_dir, valFold=3, testingHoldoutFold=4, test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # Pretrain discriminator if pretrain_disc is True
    if pretrain_disc:
        for pretrain_epoch in range(pretrain_disc_epoch):
            #train discriminator only
            for idx, volumes in enumerate(train_loader):
                real_dose = volumes[:, 0, :, :, :].unsqueeze(1).float()
                est_dose = volumes[:, 1, :, :, :].unsqueeze(1).float()
                oars = volumes[:, 2, :, :, :].unsqueeze(1).float()

                est_dose = est_dose.cuda()
                real_dose = real_dose.cuda()
                oars = oars.cuda()

                D_real = d(real_dose, est_dose, oars)
                D_fake = d(est_dose, est_dose, oars)

                D_real_loss = adv_criterion(D_real, torch.ones_like(D_real))
                D_fake_loss = adv_criterion(D_fake, torch.zeros_like(D_fake))

                D_loss = (D_real_loss + D_fake_loss) / 2

                d.zero_grad()
                d_scaler.scale(D_loss).backward(retain_graph=True)
                d_scaler.step(opt_d)
                d_scaler.update()

    # mkd
    log_path = os.path.join(save_dir, "Logs", exp_name_base, exp_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    writer = SummaryWriter(log_path)

    epoch_loop = tqdm.tqdm(range(num_epochs + 1))
    # for epoch in range(num_epochs):
    for epoch in epoch_loop:
        g.train()
        d.train()
        for idx, volumes in enumerate(train_loader):
            real_dose = volumes[:, 0, :, :, :].unsqueeze(1).float()
            est_dose = volumes[:, 1, :, :, :].unsqueeze(1).float()
            #order of stack is: [real dose; estimated dose; oars; CT; Prescription]
            if alt_condition_volume == "ED": #ED only
                alt_condition = est_dose
            elif alt_condition_volume == "CT": #prescription and ct
                alt_condition = torch.cat((volumes[:, 4, :, :, :].unsqueeze(1).float(), volumes[:, 3, :, :, :].unsqueeze(1).float()), dim=1)
            elif alt_condition_volume == "EDCT":  #ED and CT
                alt_condition = torch.cat(
                    (est_dose, volumes[:, 3, :, :, :].unsqueeze(1).float()), dim=1)
            else:
                raise Exception("Check alternative condition volume")

            alt_condition = alt_condition.cuda()
            oars = volumes[:, 2, :, :, :].unsqueeze(1).float()

            est_dose = est_dose.cuda()
            real_dose = real_dose.cuda()
            oars = oars.cuda()

            # Train Discriminator
            with torch.cuda.amp.autocast():
                D_loss, D_real_loss, D_fake_loss = getDLoss(g, d, real_dose, oars, alt_condition, est_dose, adv_criterion)

            #Update disc less often than gen if d_update_ratio > 1
            if round(d_update_ratio) >= 1:
                if epoch % round(d_update_ratio) == 0:
                    d.zero_grad()
                    d_scaler.scale(D_loss).backward(retain_graph=True)
                    d_scaler.step(opt_d)
                    d_scaler.update()
            else:
                d.zero_grad()
                d_scaler.scale(D_loss).backward(retain_graph=True)
                d_scaler.step(opt_d)
                d_scaler.update()

            # Train Generator
            with torch.cuda.amp.autocast():
                #Note the y_fake that is returned has NOT been detached
                G_loss, G_Dcomp_loss_train, voxel_loss_train, masked_voxel_loss_train, y_fake = getGLoss(g, d, real_dose, oars,
                                                                                               alt_condition, est_dose, adv_criterion,
                                                                                               voxel_criterion, alpha, beta)
            #Update gen less often than disc if g_update_ratio > 1
            if round(d_update_ratio) <= 1:
                if epoch % round(1/d_update_ratio) == 0:
                    opt_g.zero_grad()
                    g_scaler.scale(G_loss).backward()
                    g_scaler.step(opt_g)
                    g_scaler.update()
            else:
                opt_g.zero_grad()
                g_scaler.scale(G_loss).backward()
                g_scaler.step(opt_g)
                g_scaler.update()

        # save weights
        weights_path = os.path.join(save_dir, "Weights", exp_name_base, exp_name)
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)

        if epoch % 100 == 0 or epoch == num_epochs:
            torch.save(g.state_dict(),
                       os.path.join(weights_path, "GeneratorWeightsEpoch" + str(epoch) + ".pth"))
            torch.save(d.state_dict(),
                       os.path.join(weights_path, "DiscriminatorWeightsEpoch" + str(epoch) + ".pth"))

        # calculate extra losses and metrics
        y_fake = y_fake.detach()
        train_mse_loss = L2_LOSS_sum(y_fake, real_dose) / torch.numel(y_fake)
        masked_train_mse_loss = L2_LOSS_sum(y_fake * (oars > 0), real_dose * (oars > 0)) / torch.sum(oars > 0)
        fake_V20_train = getV_XfromTensor(y_fake, oars, oarCodes["lung"], 20)
        true_V20_train = getV_XfromTensor(real_dose, oars, oarCodes["lung"], 20)
        diff_train_V20 = np.mean(np.abs(true_V20_train - fake_V20_train))

        # Testing
        if epoch % log_interval == 0:
            g.eval()
            d.eval()
            G_loss_test = 0
            for test_idx, test_volumes in enumerate(test_loader):
                with torch.no_grad():
                    real_dose_test = test_volumes[:, 0, :, :, :].unsqueeze(1).float()
                    est_dose_test = test_volumes[:, 1, :, :, :].unsqueeze(1).float()
                    if alt_condition_volume == "ED": #ED only
                        alt_condition_test = est_dose_test
                    elif alt_condition_volume == "CT": #Prescription and CT
                        alt_condition_test = torch.cat((test_volumes[:, 4, :, :, :].unsqueeze(1).float(), test_volumes[:, 3, :, :, :].unsqueeze(1).float()), dim=1)
                    elif alt_condition_volume == "EDCT": #ED and CT
                        alt_condition_test = torch.cat((est_dose_test,
                                                        test_volumes[:, 3, :, :, :].unsqueeze(1).float()), dim=1)
                    else:
                        raise Exception("Check alternative condition volume")
                    oars_test = test_volumes[:, 2, :, :, :].unsqueeze(1).float()

                    alt_condition_test = alt_condition_test.cuda()
                    real_dose_test = real_dose_test.cuda()
                    oars_test = oars_test.cuda()
                    est_dose_test = est_dose_test.cuda()

                    D_loss_test, D_real_loss_test, D_fake_loss_test = getDLoss(g, d, real_dose_test, oars_test, alt_condition_test, est_dose_test, adv_criterion)

                    #Returned y_fake_test NOT been detached
                    G_loss_test, G_Dcomp_loss_test, voxel_loss_test, masked_voxel_loss_test, y_fake_test = getGLoss(g, d,
                                                                                                           real_dose_test,
                                                                                                           oars_test,
                                                                                                           alt_condition_test,
                                                                                                                    est_dose_test,
                                                                                                           adv_criterion,
                                                                                                           voxel_criterion, alpha, beta)

            G_loss_test /= (test_idx + 1)


            # calculate extra losses and metrics
            y_fake_test = g(alt_condition_test, oars_test).detach()
            test_mse_loss = L2_LOSS_sum(y_fake_test, real_dose_test) / torch.numel(y_fake_test)
            masked_test_mse_loss = L2_LOSS_sum(y_fake_test * (oars_test > 0), real_dose_test * (oars_test > 0)) / torch.sum(oars_test > 0)
            fake_V20_test = getV_XfromTensor(y_fake_test, oars_test, oarCodes["lung"], 20)
            true_V20_test = getV_XfromTensor(real_dose_test, oars_test, oarCodes["lung"], 20)
            diff_test_V20 = np.mean(np.abs(true_V20_test - fake_V20_test))

        # if not os.path.exists(os.path.join(save_dir, "Images")):
        #     os.mkdir(os.path.join(save_dir, "Images"))
        # if not os.path.exists(os.path.join(save_dir, "Images", exp_name_base)):
        #     os.mkdir(os.path.join(save_dir, "Images", exp_name_base))
        images_save_path = os.path.join(save_dir, "Images", exp_name_base, exp_name)
        if not os.path.exists(images_save_path):
            os.makedirs(os.path.join(images_save_path))

        # Saves images for all items in last batch
        if epoch % log_interval == 0:
            y_fake_test = y_fake_test.cpu().numpy()
            real_dose_test = real_dose_test.cpu().numpy()
            alt_condition_test = alt_condition_test.detach().cpu().numpy()
            oars_test = oars_test.detach().cpu().numpy()
            ct_test = test_volumes[:, 3, :, :, :].unsqueeze(1).float().detach().numpy()
            for j in range(y_fake_test.shape[0]):
                try:
                    saveImgNby3(
                        [y_fake_test[j, 0, :, :, :], real_dose_test[j, 0, :, :, :], alt_condition_test[j, 0, :, :, :]],
                        ct_test[j, 0, :, :, :],
                        os.path.join(images_save_path, str(j) + "_epoch" + str(epoch) + ".png"),
                        labels=["Fake", "Real", "Condition"])
                except:
                    print("Error saving image")

        # Logging
        writer.add_scalar('LossG/train', G_loss, epoch)
        writer.add_scalar('LossG/test', G_loss_test, epoch)
        writer.add_scalar('LossD/train', D_loss, epoch)
        writer.add_scalar('LossD/test', D_loss_test, epoch)
        writer.add_scalar('LossD_fake/train', D_fake_loss, epoch)
        writer.add_scalar('LossD_real/train', D_real_loss, epoch)
        writer.add_scalar('LossD_fake/test', D_fake_loss_test, epoch)
        writer.add_scalar('LossD_real/test', D_real_loss_test, epoch)
        writer.add_scalar('MSE/train', train_mse_loss, epoch)
        writer.add_scalar('MSE/test', test_mse_loss, epoch)
        writer.add_scalar('Masked_MSE/train', masked_train_mse_loss, epoch)
        writer.add_scalar('Masked_MSE/test', masked_test_mse_loss, epoch)
        writer.add_scalar('G_D_loss/train', G_Dcomp_loss_train, epoch)
        writer.add_scalar('G_D_loss/test', G_Dcomp_loss_test, epoch)
        writer.add_scalar('G_Voxel_loss/train', voxel_loss_train, epoch)
        writer.add_scalar('G_Voxel_loss/test', voxel_loss_test, epoch)
        writer.add_scalar('Masked_G_Voxel_loss/train', masked_voxel_loss_train, epoch)
        writer.add_scalar('Masked_G_Voxel_loss/test', masked_voxel_loss_test, epoch)
        writer.add_scalar('Diff_V20/train', diff_train_V20, epoch)
        writer.add_scalar('Diff_V20/test', diff_test_V20, epoch)


    writer.add_hparams(
        {"epochs": num_epochs, "alpha": alpha, "beta": beta, "loss_type": loss_type, "d_update_ratio": d_update_ratio,
         "attention": generator_attention, "batch_size": batch_size, "g_lr": g_lr, "d_lr": g_lr, "condition": alt_condition_volume,
         "adv_loss_type": adv_loss_type, "pretrain_disc": pretrain_disc, "pretrain_disc_epoch": pretrain_disc_epoch,},
        {"hparam/last_mse_loss_test": test_mse_loss, "hparam/last_g_loss_test": G_loss_test,
         "hparam/last_d_loss_test": D_loss_test},
        run_name=log_path)  # <- see here
    writer.close()

if __name__ == '__main__':
    print(torch.cuda.is_available())
    config = load_config("conf.yml")
    data_dir = config.DATA_DIR
    patientList_dir = config.PATIENT_LIST_DIR
    save_dir = config.SAVE_DIR

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_epochs", type=int)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument('--attention', action=argparse.BooleanOptionalAction)
        parser.add_argument("--exp_name", type=str)
        parser.add_argument("--condition", type=str)
        parser.add_argument("--alpha", type=float)
        parser.add_argument("--beta", type=float)
        parser.add_argument("--loss_type", type=str)
        parser.add_argument("--dur", type=float)
        parser.add_argument("--g_lr", type=float)
        parser.add_argument("--d_lr", type=float)
        parser.add_argument("--adv_loss_type", type=str)
        parser.add_argument("--log_interval", type=int)
        parser.add_argument("--pretrain_disc", action=argparse.BooleanOptionalAction)
        parser.add_argument("--pretrain_disc_epochs", type=int)

        #parser.add_argument("--recalc_fake", action=argparse.BooleanOptionalAction)

        args = parser.parse_args()

        num_epochs = args.num_epochs
        batch_size = args.batch_size
        generator_attention = args.attention
        exp_name_base = args.exp_name
        alt_condition_volume = args.condition
        alphas = [args.alpha]
        betas = [args.beta]
        loss_types = [args.loss_type]
        d_update_ratios = [args.dur]
        g_lrs = [args.g_lr]
        d_lrs = [args.d_lr]
        #recalc_fake = args.recalc_fake
        adv_loss_types = [args.adv_loss_type]
        log_interval = args.log_interval
        pretrain_disc = args.pretrain_disc
        pretrain_disc_epochs = [args.pretrain_disc_epochs]

    else:
        num_epochs = config.NUM_EPOCHS
        batch_size = config.BATCH_SIZE
        generator_attention = config.GENERATOR_ATTENTION
        exp_name_base = config.EXP_NAME
        alt_condition_volume = config.ALT_CONDITION_VOLUME
        alphas = config.ALPHA
        betas = config.BETA
        loss_types = config.LOSS_TYPE
        d_update_ratios = config.D_UPDATE_RATIO
        g_lrs = config.G_LR
        d_lrs = config.D_LR
        #recalc_fake = config.RECALC_FAKE
        adv_loss_types = config.ADV_LOSS_TYPE
        log_interval = config.LOG_INTERVAL
        pretrain_disc = config.PRETRAIN_DISC
        pretrain_disc_epochs = config.PRETRAIN_DISC_EPOCHS

    runNum = 0
    for alpha in alphas:
        for beta in betas:
            for loss_type in loss_types:
                for d_update_ratio in d_update_ratios:
                    for g_lr in g_lrs:
                        for d_lr in d_lrs:
                            for adv_loss_type in adv_loss_types:
                                for pretrain_disc_epoch in pretrain_disc_epochs:
                                    params = {
                                        "num_epochs": num_epochs,
                                        "alpha": alpha,
                                        "beta": beta,
                                        "loss_type": loss_type,
                                        "d_update_ratio": d_update_ratio,
                                        "batch_size": batch_size,
                                        "generator_attention": generator_attention,
                                        "alt_condition_volume": alt_condition_volume,
                                        "g_lr": float(g_lr),
                                        "d_lr": float(d_lr),
                                        #"recalc_fake": recalc_fake,
                                        "adv_loss_type": adv_loss_type,
                                        "log_interval": log_interval,
                                        "pretrain_disc": pretrain_disc,
                                        "pretrain_disc_epoch": pretrain_disc_epoch,
                                    }

                                    exp_name = f'dLR={d_lr}_Lo={loss_type}_gLR={g_lr}_Alp={alpha}_Beta={beta}_DUR={d_update_ratio}_BtSiz={batch_size}_Att={generator_attention}_Con={alt_condition_volume}_AdvLo={adv_loss_type}_PrTD={pretrain_disc}_PrTDEp={pretrain_disc_epoch}'
                                    print(params, exp_name)
                                    train(data_dir, patientList_dir, save_dir, exp_name_base, exp_name, params)

                                    runNum += 1

# C:\Users\wanged\Anaconda3\envs\LungGan\Scripts\t