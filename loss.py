import torch
from torch import nn

class GDL(torch.nn.Module):
    def init(self):
        super(GDL, self).init()
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

class V20Loss(torch.nn.Module):
    def init(self):
        super(V20Loss, self).init()
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