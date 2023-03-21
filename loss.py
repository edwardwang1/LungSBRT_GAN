import numpy as np
import torch
from torch import nn
import lpips
from einops import rearrange

class GDL(nn.Module):
    def __init__(self):
        super(GDL, self).__init__()
    def forward(self,pred, target):
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection) / (A_sum + B_sum ))


def getV_X(doseVolume, oarVolume, oarCode, doseTarget):
    oar = oarVolume == oarCode
    greaterThanDose = doseVolume > doseTarget
    return (greaterThanDose * oar).sum()/oar.sum() * 100

class V20Loss(nn.Module):
    def __init__(self):
        super(V20Loss, self).__init__()
    def forward(self, fake, real, oars):
        # Lung code is 1
        lungCode = 1
        # Threshold is 20

        # Get the V20 for the fake and real

        lungMask = oars == lungCode
        lungVolume = lungMask.sum()

        fakeV20Volume = ((fake > 20) * lungMask).sum()
        realV20Volume = ((real > 20) * lungMask).sum()

        # Return the difference
        return torch.abs((fakeV20Volume - realV20Volume) / lungVolume)

class LinearScaler(nn.Module):
    def __init__(self):
        super(LinearScaler, self).__init__()

    def forward(self, x, min, max, selfScale=False):
        # x is a tensor of any shape
        if selfScale:
            min = torch.min(x)
            max = torch.max(x)
        scaled_x = (x - min) / (max - min) * 2 - 1
        return scaled_x

class LPIPSLoss(torch.nn.Module):
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        self.loss_fn = lpips.LPIPS(net='vgg').cuda()
        self.scaler = LinearScaler() #normalize fake and real to be between -1 and 1


    def getSliceStack(self, tensor, index,axis, extension):
        if axis == 0:
            first_slice = tensor[index-extension, :, :]
            middle_slice = tensor[index, :, :]
            last_slice = tensor[index+extension, :, :]
        elif axis == 1:
            first_slice = tensor[:, index-extension, :]
            middle_slice = tensor[:, index, :]
            last_slice = tensor[:, index+extension, :]
        elif axis == 2:
            first_slice = tensor[:, :, index-extension]
            middle_slice = tensor[:, :, index]
            last_slice = tensor[:, :, index+extension]


        stack = torch.stack([first_slice, middle_slice, last_slice], dim=0)
        #print(stack.shape)
        return stack


    def getAnalaysisVolumes(self, fake, real):
        # Dimensions are (batch, channel, width, height, depth)


        fake = fake.squeeze(0)
        real = real.squeeze(0)

        # should have dimensions (width, height, depth)
        flat_index = torch.argmax(real)
        h_index = int(torch.div(flat_index, (real.shape[1] * real.shape[2]), rounding_mode='floor'))
        w_index = int(torch.div(flat_index % (real.shape[1] * real.shape[2]), real.shape[2],
                                rounding_mode='floor'))
        d_index = int(flat_index % real.shape[2])



        # Ensure that indicies are at least Steps away from the edge
        steps = 2

        h_index = max(h_index, steps * 2)
        h_index = min(h_index, real.shape[0] - 3)
        w_index = max(w_index, steps * 2)
        w_index = min(w_index, real.shape[1] - 3)
        d_index = max(d_index, steps * 2)
        d_index = min(d_index, real.shape[2] - 3)

        # Remove the channel dimension
        min_real = torch.min(real)
        max_real = torch.max(real)


        # fake_ax = self.scaler(fake[h_index - 3:h_index + 3, :, :].unsqueeze(0), min_real, max_real, selfScale=False)
        # real_ax = self.scaler(real[h_index - 3:h_index + 3, :, :].unsqueeze(0), min_real, max_real, selfScale=False)
        #
        # fake_cor = rearrange(self.scaler(fake[:, w_index - 3: w_index + 3, :].unsqueeze(0), min_real, max_real, selfScale=False), 'c h w d -> c w h d')
        # real_cor = rearrange(self.scaler(real[:, w_index - 3: w_index + 3, :].unsqueeze(0), min_real, max_real, selfScale=False), 'c h w d -> c w h d')
        #
        # fake_sag = rearrange(self.scaler(fake[:, :, d_index - 3: d_index + 3].unsqueeze(0), min_real, max_real, selfScale=False), 'c h w d -> c d w h')
        # real_sag = rearrange(self.scaler(real[:, :, d_index - 3: d_index + 3].unsqueeze(0), min_real, max_real, selfScale=False), 'c h w d -> c d w h')


        fake_ax = self.scaler(self.getSliceStack(fake, h_index, 0, steps).unsqueeze(0), min_real, max_real, selfScale=False)
        real_ax = self.scaler(self.getSliceStack(real, h_index, 0, steps).unsqueeze(0), min_real, max_real, selfScale=False)

        fake_cor = self.scaler(self.getSliceStack(fake, w_index, 1, steps).unsqueeze(0), min_real, max_real, selfScale=False)
        real_cor = self.scaler(self.getSliceStack(real, w_index, 1, steps).unsqueeze(0), min_real, max_real, selfScale=False)

        fake_sag = self.scaler(self.getSliceStack(fake, d_index, 2, steps).unsqueeze(0), min_real, max_real, selfScale=False)
        real_sag = self.scaler(self.getSliceStack(real, d_index, 2, steps).unsqueeze(0), min_real, max_real, selfScale=False)

        #fake_cor = rearrange(self.scaler(self.getSliceStack(fake, w_index, 1, steps).unsqueeze(0), min_real, max_real, selfScale=False), 'b h w d -> b w h d')
        #real_cor = rearrange(self.scaler(self.getSliceStack(real, w_index, 1, steps).unsqueeze(0), min_real, max_real, selfScale=False), 'b h w d -> b w h d')

        #fake_sag = rearrange(self.scaler(self.getSliceStack(fake, d_index, 2, steps).unsqueeze(0), min_real, max_real, selfScale=False), 'b h w d -> b d w h')
        #real_sag = rearrange(self.scaler(self.getSliceStack(real, d_index, 2, steps).unsqueeze(0), min_real, max_real, selfScale=False), 'b h w d -> b d w h')


        #print(fake_ax.shape, real_ax.shape, fake_cor.shape, real_cor.shape, fake_sag.shape, real_sag.shape)
        return fake_ax, real_ax, fake_cor, real_cor, fake_sag, real_sag

    def forward(self, fake, real):
        # Get the LPIPS loss

        acc_perceptual_loss = 0
        num_batches = fake.shape[0]

        #Iterate through batches
        for i in range(num_batches):
            #Get min and max of real for scaling

            fake_ax, real_ax, fake_cor, real_cor, fake_sag, real_sag = self.getAnalaysisVolumes(fake[i], real[i])

            acc_perceptual_loss += self.loss_fn.forward(fake_ax, real_ax)
            acc_perceptual_loss += self.loss_fn.forward(fake_cor, real_cor)
            acc_perceptual_loss += self.loss_fn.forward(fake_sag, real_sag)

        return acc_perceptual_loss / (num_batches * 3.)

if __name__ == '__main__':
    #Testing LIPIPS loss
    fake = torch.rand(1, 1, 64, 64, 64)
    fake[0, 0, 16, 4, 59] = 9999
    real = torch.rand(16, 1, 64, 64, 64)

    loss_fn = LPIPSLoss()
    loss = loss_fn(fake.cuda(), real.cuda())
    print(loss)
    print(fake.flatten()[65851])