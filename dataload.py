from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import torch as t
import numpy as np
import os

device = t.device('cuda' if t.cuda.is_available() else 'cpu')


# load HSI
def Load_HS(root):
    data = loadmat(os.path.join(root, "norm_urban.mat"))['normalized_m'] / 1.0
    return t.tensor(data).to(device)

# load HSI
def Load_tatget_HS(root):
    data = loadmat(os.path.join(root, "target_c.mat"))['T_new']/1.0
    return t.tensor(data).to(device)

# load HSI
def Load_background_HS(root):
    data = loadmat(os.path.join(root, "background_c.mat"))['B_new']/1.0
    return t.tensor(data).to(device)

# load Db
def Load_Db(root):
    Db = loadmat(root)['Db']
    return t.tensor(Db).to(device)


# load Db0
def Load_Db0(root):
    Db = loadmat(os.path.join(root, "Db0.mat"))['Db0']
    return t.tensor(Db).to(device)


# load Dt
def Load_Dt(root):
    Dt = loadmat(os.path.join(root, "Dt.mat"))['Dt']
    return t.tensor(Dt).to(device)


# load background sample
class Background_Dataset(Dataset):
    def __init__(self, root):
        super(Background_Dataset, self).__init__()
        self.Y = loadmat(os.path.join(root, "background_sample.mat"))['b_s_w']

    def __getitem__(self, item):
        Y_out = self.Y[:, item].reshape(-1, 1)
        return Y_out.to(device)

    def __len__(self):
        return self.Y.shape[1]

    def getall(self):
        Y_out = np.expand_dims(self.Y.T, -1)    # 转置+更改维度
        return t.tensor(Y_out).to(device)


# load target sample
class Target_Dataset(Dataset):
    def __init__(self, root):
        super(Target_Dataset, self).__init__()
        self.I = loadmat(os.path.join(root, "target_sample.mat"))['t_s_w']

    def __getitem__(self, item):
        I_out = self.I[:, item].reshape(-1, 1)
        return I_out.to(device)

    def __len__(self):
        return self.I.shape[1]

    def getall(self):
        I_out = np.expand_dims(self.I.T, -1)    # 转置+更改维度
        return t.tensor(I_out).to(device)


if __name__ == "__main__":
    root = ""
    data = Background_Dataset(root)
    print(data.__len__())
    print(data.__getitem__(0).shape)

    # Db0 = Load_Db0(root)
    # print(Db0)

    # Dt = Load_Dt(root)
    # print(Dt)

    # data = Target_Dataset(root)
    # print(data.__len__())
    # print(data.__getitem__(0).shape)

    # data = Load_HS(root)
    # print(data.shape)