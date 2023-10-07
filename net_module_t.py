import torch as t
import torch.nn as nn
import torch.nn.functional as F
from dataload import Load_Dt
import numpy as np
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

def beta0_get(Dt, I):
    Dt_inv = np.linalg.pinv(Dt.detach().cpu().numpy())
    Dt_inv = t.tensor(Dt_inv).to(device)
    beta0 = t.matmul(Dt_inv, I)
    return beta0

def weight_init(root, I):
    Dt = Load_Dt(root).type(t.float)
    beta0 = beta0_get(Dt, I)
    Dt_t = t.transpose(Dt, dim0=-1, dim1=-2)
    temp = beta0 + t.matmul(Dt_t, I)
    # change the shape of alpha and temp
    beta0 = t.transpose(beta0, 0, 1).squeeze(-1)
    temp = t.transpose(temp, 0, 1).squeeze(-1)
    temp_inv = np.linalg.pinv(temp.cpu())
    weight = t.matmul(beta0, t.tensor(temp_inv).to(device))
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
        # print(x_128.size(), xl_128.size())
        y_256 = t.cat((x_128, xl_128), dim=-2)
        # print(y_256.size())
        yl_64 = self.conv_256_64(y_256)
        y_128 = t.cat((x_64, yl_64), dim=-2)
        y_64 = self.conv_128_64(y_128)
        y = self.conv_out(y_64)
        return y


# Generate alpha from S
class Net_S2a(nn.Module):
    def __init__(self, bands, root, I):
        super(Net_S2a, self).__init__()
        # 参数 mu 设置为可学习的参数
        self.mu = nn.Parameter(t.FloatTensor(1), requires_grad=True)
        # 利用 乘法系数
        self.myweight = nn.Parameter(data=weight_init(root, I), requires_grad=True)
        # 利用卷积 模拟 矩阵乘
        self.conv_a = nn.Sequential(
            nn.Conv1d(in_channels=bands, out_channels=bands, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
            # nn.Softmax(dim=-2)
        )

    def forward(self, S, Y, Db):
        Db_t = t.transpose(Db, dim0=-1, dim1=-2)
        temp = (self.mu * S) + t.matmul(Db_t, Y)
        a = F.conv1d(temp, self.myweight)
        a = F.relu(a)
        return a

class repeat_a_block(nn.Module):
    def __init__(self, bands, root, I):
        super(repeat_a_block, self).__init__()
        self.bands = bands
        self.neta2S = Net_a2S(bands)
        self.netS2a = Net_S2a(bands, root, I)

    def forward(self, alpha, Y, Db):
        S = self.neta2S(alpha)
        alpha_new = self.netS2a(S, Y, Db)
        return alpha_new, S

class Net_main_t(nn.Module):
    def __init__(self, bands, times, root, I):
        super(Net_main_t, self).__init__()
        self.bands = bands
        self.times = times
        self.repeat_a_block = repeat_a_block(bands, root, I)

    def forward(self, alpha, Y, Db):
        for i in range(self.times):
            (alpha_new, S) = self.repeat_a_block(alpha, Y, Db)
            alpha = alpha_new

        return alpha, S

class Net_test_main_t(nn.Module):
    def __init__(self, bands, times, root, I):
        super(Net_test_main_t, self).__init__()
        self.bands = bands
        self.times = times
        self.repeat_a_block = repeat_a_block(bands, root, I)

    def forward(self, alpha, Y, Db):
        for i in range(self.times):
            (alpha_new, S) = self.repeat_a_block(alpha, Y, Db)
            alpha = alpha_new

        return alpha, S


if __name__ == "__main__":
    # test_Net_a2S()
    # test_Net_S2a()
    # test_Net_main()
    pass
