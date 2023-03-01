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

    def forward(self, x, min, max):
        # x is a tensor of any shape
        scaled_x = (x - min) / (max - min) * 2 - 1
        return scaled_x

class LPIPSLoss(torch.nn.Module):
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        self.loss_fn = lpips.LPIPS(net='vgg').cuda()
        self.scaler = LinearScaler()

    def forward(self, fake, real):
        # Get the LPIPS loss
        #normalize fake and real to be between -1 and 1

        #Dimensions are (batch, channel, width, height, depth)
        # Get index of fake with largest value
        #Remove channel dimension
        fake = fake.squeeze(1)
        real = real.squeeze(1)

        acc_perceptual_loss = 0
        num_batches = fake.shape[0]

        #Iterate through batches
        for i in range(num_batches):
            curr_tensor = fake[i] #should have dimensions (width, height, depth)
            flat_index = torch.argmax(curr_tensor)
            #h_index = flat_index // (curr_tensor.shape[1] * curr_tensor.shape[2])
            h_index = int(torch.div(flat_index, (curr_tensor.shape[1] * curr_tensor.shape[2]), rounding_mode='floor'))
            w_index = int(torch.div(flat_index % (curr_tensor.shape[1] * curr_tensor.shape[2]), curr_tensor.shape[2], rounding_mode='floor'))
            d_index = int(flat_index % curr_tensor.shape[2])

            #Ensure that indicies are at least 3 away from the edge
            h_index = max(h_index, 3)
            h_index = min(h_index, curr_tensor.shape[0] - 3)
            w_index = max(w_index, 3)
            w_index = min(w_index, curr_tensor.shape[1] - 3)
            d_index = max(d_index, 3)
            d_index = min(d_index, curr_tensor.shape[2] - 3)

            #Get min and max of real for scaling
            min_real = torch.min(real[i])
            max_real = torch.max(real[i])

            acc_perceptual_loss += self.loss_fn.forward(self.scaler(fake[i, h_index - 1:h_index + 2, :, :].unsqueeze(0), min_real, max_real),
                                                        self.scaler(real[i, h_index - 1:h_index + 2, :, :].unsqueeze(0), min_real, max_real))
            acc_perceptual_loss += self.loss_fn.forward(rearrange(self.scaler(fake[i, :, w_index - 1: w_index + 2, :].unsqueeze(0), min_real, max_real), 'c h w d -> c w h d'),
                                                        (rearrange(self.scaler(real[i, :, w_index - 1: w_index + 2, :].unsqueeze(0), min_real, max_real), 'c h w d -> c w h d')))
            acc_perceptual_loss += self.loss_fn.forward(rearrange(self.scaler(fake[i, :, :, d_index - 1: d_index + 2].unsqueeze(0), min_real, max_real), 'c h w d -> c d w h'),
                                                        (rearrange(self.scaler(real[i, :, :, d_index - 1: d_index + 2].unsqueeze(0), min_real, max_real), 'c h w d -> c d w h')))

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