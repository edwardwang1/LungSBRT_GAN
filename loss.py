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