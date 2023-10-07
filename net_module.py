import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataload import Load_Db0
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# For each batch splice Y
# 将 Y(batchsize*c*1)-->Y(c*batchisze)
def cat_Y(Y):
    Y = Y.squeeze(dim=-1)
    Y_cat = t.transpose(Y, dim0=-1, dim1=-2)
    return Y_cat

# For each batch splice alpha
# 将 alpha(batchsize*k*1)-->alpha(k*batchisze)hz
def cat_alpha(alpha):
    alpha = alpha.squeeze(dim=-1)
    alpha_cat = t.transpose(alpha, dim0=-1, dim1=-2)
    return alpha_cat

# Find the inverse of the matrix
def alpha0_get(Db, Y):
    Db_inv = np.linalg.pinv(Db.detach().cpu().numpy())
    Db_inv = t.tensor(Db_inv).to(device)
    alpha0 = t.matmul(Db_inv, Y)
    return alpha0

# Generate a new dictionary DB from Alpha and Y
def Db_out(alpha, Y):
    alpha = cat_alpha(alpha)
    Y = cat_Y(Y)
    alpha_t = t.transpose(alpha, dim0=-1, dim1=-2)
    out1 = t.matmul(Y, alpha_t)
    out2 = np.linalg.pinv(t.matmul(alpha, alpha_t).detach().cpu().numpy())
    out2 = t.tensor(out2).to(device)
    out = t.matmul(out1, out2)
    return out

def weight_init(root, Y):
    Db0 = Load_Db0(root).type(t.float)
    alpha0 = alpha0_get(Db0, Y)
    Db_t = t.transpose(Db0, dim0=-1, dim1=-2)
    temp = alpha0 + t.matmul(Db_t, Y)
    # change the shape of alpha and temp
    alpha0 = t.transpose(alpha0, 0, 1).squeeze(-1)
    temp = t.transpose(temp, 0, 1).squeeze(-1)
    temp_inv = np.linalg.pinv(temp.cpu())
    weight = t.matmul(alpha0, t.tensor(temp_inv).to(device))

    return weight.unsqueeze(-1)


def ConvB(inputc, outputc):
    conv = nn.Sequential(
        nn.Conv1d(in_channels=inputc, out_channels=outputc, kernel_size=1, stride=1, padding=0),
        nn.ReLU()
    )
    return conv

# Generate s from alpha
class Net_a2S(nn.Module):
    def __init__(self, bands):
        super(Net_a2S, self).__init__()
        self.conv_in = ConvB(bands, 64)
        self.conv_64_128 = ConvB(64, 128)
        self.conv_128_256 = ConvB(128, 256)
        self.conv_256_128 = ConvB(256, 128)
        self.conv_256_64 = ConvB(256, 64)
        self.conv_128_64 = ConvB(128, 64)
        self.conv_out = ConvB(64, bands)

    def forward(self, x):
        x_64 = self.conv_in(x)
        x_128 = self.conv_64_128(x_64)
        x_256 = self.conv_128_256(x_128)
        xl_128 = self.conv_256_128(x_256)
        y_256 = t.cat((x_128, xl_128), dim=-2)
        yl_64 = self.conv_256_64(y_256)
        y_128 = t.cat((x_64, yl_64), dim=-2)
        y_64 = self.conv_128_64(y_128)
        y = self.conv_out(y_64)
        return y

# Generate alpha from S
class Net_S2a(nn.Module):
    def __init__(self, bands, root, Y0):
        super(Net_S2a, self).__init__()
        # 参数 mu 设置为可学习的参数
        self.mu = nn.Parameter(t.FloatTensor([1]), requires_grad=True)

        # 利用 乘法系数
        self.myweight = nn.Parameter(data=weight_init(root, Y0), requires_grad=True)

        # # 利用卷积 模拟 矩阵乘
        self.conv_a = nn.Sequential(
            nn.Conv1d(in_channels=bands, out_channels=bands, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, S, Y, Db):
        Db_t = t.transpose(Db, dim0=-1, dim1=-2)
        temp = (self.mu * S) + t.matmul(Db_t, Y)
        a = F.conv1d(temp, self.myweight)
        a = F.relu(a)
        return a

class repeat_a_block(nn.Module):
    def __init__(self, bands, root, Y0):
        super(repeat_a_block, self).__init__()
        self.bands = bands
        self.neta2S = Net_a2S(bands)
        self.netS2a = Net_S2a(bands, root, Y0)

    def forward(self, alpha, Y, Db):
        S = self.neta2S(alpha)
        alpha_new = self.netS2a(S, Y, Db)
        return alpha_new, S

class Net_main(nn.Module):
    def __init__(self, bands, times, root, Y0):
        super(Net_main, self).__init__()
        self.bands = bands
        self.times = times
        self.repeat_a_block = repeat_a_block(bands, root, Y0)

    def forward(self, alpha, Y, Db, epoch):
        for i in range(self.times):
            (alpha_new, S) = self.repeat_a_block(alpha, Y, Db)
            Db_new = Db_out(alpha_new, Y)
            alpha = alpha_new
            Db = Db_new
        return alpha, Db, S

class Net_test_main(nn.Module):
    def __init__(self, bands, root, Y0):
        super(Net_test_main, self).__init__()
        self.bands = bands
        self.repeat_a_block = repeat_a_block(bands, root, Y0)

    def forward(self, alpha, Y, Db):
        (alpha_new, S) = self.repeat_a_block(alpha, Y, Db)
        return alpha_new, S

if __name__ == "__main__":
    # test_Net_a2S()
    # test_Net_S2a()
    # test_Net_main()
    pass
