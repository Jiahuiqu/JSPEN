import torch as t
import torch.nn as nn
import scipy.io as scio
from train import Db0

class Loss1(nn.Module):
    def __init__(self):
        super(Loss1, self).__init__()

    def forward(self, Y, Db, alpha):
        loss0 = Y-t.matmul(Db, alpha)
        loss = t.norm(loss0, p=1)
        return loss/Y.size()[0]

class Loss2(nn.Module):
    def __init__(self):
        super(Loss2, self).__init__()

    def forward(self, alpha):
        loss = t.norm(alpha, p=1)
        return loss/alpha.size()[0]

class Loss_all(nn.Module):
    def __init__(self):
        super(Loss_all, self).__init__()

    def forward(self, Y, Db, alpha):
        loss0 = Y-t.matmul(Db, alpha)
        loss1 = t.norm(loss0, p=1)
        loss2 = t.norm(alpha, p=1)
        loss3 = t.norm((Db - Db0), p=1)
        loss = loss1 + 0.1*loss2 + 0.1*loss3
        return loss/Y.size()[0], loss1/Y.size()[0], loss2/Y.size()[0], loss3/Y.size()[0]

class Loss_all_t(nn.Module):
    def __init__(self):
        super(Loss_all_t, self).__init__()

    def forward(self, I, Dt, beta, S):
        loss0 = I-t.matmul(Dt, beta)
        loss1 = t.norm(loss0, p=1)
        loss2 = t.norm(beta, p=1)
        loss = loss1 + 0.1*loss2
        return loss/I.size()[0], loss1/I.size()[0], loss2/I.size()[0]
