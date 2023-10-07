import torch as t
import torch.nn as nn
import random
import scipy.io as scio
from torch import optim
from torch.utils.data import DataLoader
from net_module import Net_test_main
from net_module_t import Net_main_t, Net_test_main_t
from dataload import Load_HS, Load_Dt, Load_Db, Load_Db0, Load_tatget_HS, Load_background_HS
import numpy as np
import lossfunc

k = 40
k_t = 40
c = 207

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# Find the inverse of the matrix
def alpha0_get(Db, Y):
    Db_inv = np.linalg.pinv(Db.detach().cpu().numpy())
    Db_inv = t.tensor(Db_inv).to(device)
    alpha0 = t.matmul(Db_inv, Y)
    return alpha0

def beta0_get(Dt, I):
    Dt_inv = np.linalg.pinv(Dt.detach().cpu().numpy())
    Dt_inv = t.tensor(Dt_inv).to(device)
    beta0 = t.matmul(Dt_inv, I)
    return beta0

def loss_func2(Y, Db, alpha):
    lossfun1 = lossfunc.Loss1()
    lossfun2 = lossfunc.Loss2()
    loss1 = lossfun1(Y, Db, alpha)
    loss2 = lossfun2(alpha)
    loss = loss1 + 0.1*loss2
    return loss

def test(root, N):
    # data load
    data = Load_HS(root).type(t.float)
    Dt = Load_Dt(root).type(t.float)
    Db = Load_Db("./dataout/Db"+str(N)+".mat").type(t.float)
    Y0 = t.ones(5768, c, 1)
    I = t.ones(100, c, 1)

    data = data.to(device)
    Dt = Dt.to(device)
    Db = Db.to(device)
    Y0 = Y0.to(device)
    I = I.to(device)

    data_out = t.zeros(data.size()[0], data.size()[1])
    data_a = t.zeros(data.size()[0], data.size()[1])
    data_b = t.zeros(data.size()[0], data.size()[1])
    alpha_all = t.zeros(data.size()[0], data.size()[1], k)
    beta_all = t.zeros(data.size()[0], data.size()[1], k_t)
    loss_b_all = t.zeros(data.size()[0], data.size()[1])
    loss_t_all = t.zeros(data.size()[0], data.size()[1])

    # model load
    model_b = Net_test_main(k, root, Y0)
    model_b.to(device).eval()
    model_t = Net_test_main_t(k_t, 1, root, I)
    model_t.to(device).eval()
    model_b.load_state_dict(t.load("./dataout/background"+str(N)+".pkl"))
    model_t.load_state_dict(t.load("./dataout/target.pkl"))
    loss = nn.L1Loss()

    for i in range(data.size()[0]):
        for j in range(data.size()[1]):
            vector_data = data[i, j, :].reshape(-1, 1)
            alpha0 = alpha0_get(Db, vector_data)
            beta0 = beta0_get(Dt, vector_data)
            (alpha, S_a) = model_b(alpha0, vector_data, Db)
            (beta, S) = model_t(beta0, vector_data, Dt)

            loss_b0 = loss(vector_data, t.matmul(Db, alpha))
            loss_t0 = loss(vector_data, t.matmul(Dt, beta))

            alpha_norm0 = t.norm(alpha, p=0)
            beta_norm0 = t.norm(beta, p=0)
            data_a[i, j] = alpha_norm0
            data_b[i, j] = beta_norm0
            loss_b_all[i, j] = loss_b0
            loss_t_all[i, j] = loss_t0
            alpha_all[i, j, :] = alpha.reshape(1, 1, k)
            beta_all[i, j, :] = beta.reshape(1, 1, k_t)
            if alpha_norm0 <= beta_norm0:
                data_out[i, j] = 1

    scio.savemat("./dataout/loss_t_all"+str(N)+".mat", {'loss_t_all': loss_t_all.detach().numpy()})
    scio.savemat("./dataout/loss_b_all"+str(N)+".mat", {'loss_b_all': loss_b_all.detach().numpy()})
    # scio.savemat("./dataout/alpha_test.mat", {'alpha_test': alpha_all.detach().numpy()})
    # scio.savemat("./dataout/beta_test.mat", {'beta_test': beta_all.detach().numpy()})
    # scio.savemat("./dataout/data_out.mat", {'data_out': data_out.numpy()})
    # scio.savemat("./dataout/data_a.mat", {'data_a': data_a.detach().numpy()})
    # scio.savemat("./dataout/data_b.mat", {'data_b': data_b.detach().numpy()})


if __name__ == "__main__":
    root = "./dataset/base"
    test(root, 17)
    # for i in range(200):
    #     test(root, i)
    #     print(i)
