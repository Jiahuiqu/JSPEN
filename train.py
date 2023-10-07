import torch as t
import torch.nn as nn
import random
import scipy.io as scio
from torch import optim
from net_module import Net_main
from net_module_t import Net_main_t
import numpy as np
from dataload import Load_Db0, Load_Dt, Background_Dataset, Target_Dataset

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# 超参数
batchsize_b = 64
batchsize_t = 4
learningrate = 0.00001
learningrate_t = 0.000001
epochs = 200
epochs_t = 20000
k = 40
k_t = 40
c = 207

Root_Db0 = "./dataset/base"
Db0 = Load_Db0(Root_Db0).type(t.float)

def setup_seed(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.deterministic = True

# Find the inverse of the matrix
def alpha0_get(Db, Y):
    Db_inv = t.pinverse(Db.detach())
    alpha0 = t.matmul(Db_inv, Y)
    return alpha0

def beta0_get(Dt, I):
    Dt_inv = np.linalg.pinv(Dt.detach().cpu().numpy())
    Dt_inv = t.tensor(Dt_inv).to(device)
    beta0 = t.matmul(Dt_inv, I)
    return beta0

# For each batch splice Y
# 将 Y(batchsize*c*1)-->Y(c*batchisze)
def cat_Y(Y):
    Y = Y.squeeze(dim=-1)
    Y_cat = t.transpose(Y, dim0=-1, dim1=-2)
    return Y_cat

# For each batch splice alpha
# 将 alpha(batchsize*k*1)-->alpha(k*batchisze)
def cat_alpha(alpha):
    alpha = alpha.squeeze(dim=-1)
    alpha_cat = t.transpose(alpha, dim0=-1, dim1=-2)
    return alpha_cat

def train_background(root):
    # data load
    data = Background_Dataset(root)
    Db0 = Load_Db0(root).type(t.float)
    Y = data.getall()
    Y0 = Y.type(t.float)
    best_loss = 1000
    loss_mat = []
    alpha_mat = np.zeros((epochs, 6503, k))
    S_mat = np.zeros((epochs, 6503, k))
    Db_mat = np.zeros((epochs, c, k))

    # model init
    model = Net_main(k, 5, root, Y0)
    model = model.to(device)
    optimization = optim.Adam(model.parameters(), lr=learningrate)
    loss = nn.L1Loss()

    # train
    for epoch in range(epochs):
        Y = data.getall()
        Y = Y.type(t.float)
        Db = Db0
        alpha = alpha0_get(Db, Y)
        model.train()
        (alpha, Db, S) = model(alpha, Y, Db, epoch)

        loss0 = loss(Y, t.matmul(Db, alpha))
        loss1 = loss(alpha, t.zeros_like(alpha))
        loss2 = loss(Db, Db0)
        loss3 = loss(alpha, S)

        if epoch < 100:
            loss_all = loss0 + 0.01*loss1 + 1*loss2 + 0.01*loss3
        else:
            loss_all = loss0 + 0.01*loss1 + 0*loss2 + 0.01*loss3

        optimization.zero_grad()
        loss_all.backward()
        optimization.step()


        if loss_all <= best_loss:
            best_loss = loss_all
            scio.savemat("./dataout/Db.mat", {'Db': Db.detach().cpu().numpy()})
            t.save(model.state_dict(), './dataout/background.pkl')

        if epoch % 1 == 0:
            scio.savemat("./dataout/Db"+str(epoch)+".mat", {'Db': Db.detach().cpu().numpy()})
            t.save(model.state_dict(), './dataout/background'+str(epoch)+'.pkl')
            print('save checkpoint', epoch)

        print('epoch:'+str(epoch), '--best_loss:'+str(best_loss.data), '--loss:'+str(loss_all.data),
              '--loss0:'+str(loss0.data), '--loss2:'+str(loss2.data), '--loss3:'+str(loss3.data))
        loss_mat.append([loss_all.detach().cpu(), loss0.detach().cpu(), loss1.detach().cpu(), loss2.detach().cpu(), loss3.detach().cpu()])
        alpha_mat[epoch, :, :] = alpha.squeeze(-1).detach().cpu()
        S_mat[epoch, :, :] = S.squeeze(-1).detach().cpu()
        Db_mat[epoch, :, :] = Db.detach().cpu()


    # loss_mat = np.array(loss_mat)   # list->numpy
    # scio.savemat("./dataout/loss_mat_b.mat", {
    #     'loss': loss_mat[:, 0],
    #     'loss0': loss_mat[:, 1],
    #     'loss1': loss_mat[:, 2],
    #     'loss2': loss_mat[:, 3],
    #     'loss3': loss_mat[:, 4]})  # save loss

    scio.savemat("./dataout/alpha_mat.mat", {'alpha_mat': alpha_mat})
    scio.savemat("./dataout/S_mat.mat", {'S_mat': S_mat})
    scio.savemat("./dataout/Db_mat.mat", {'Db_mat': Db_mat})

def train_target(root):
    # data load
    data = Target_Dataset(root)
    Dt = Load_Dt(root).type(t.float)
    I = data.getall()
    I = I.type(t.float)
    loss = nn.L1Loss()
    best_loss = 1000
    beta_mat = np.zeros((epochs_t, I.size()[0], k_t))
    S_mat = np.zeros((epochs_t,  I.size()[0], k_t))

    model = Net_main_t(k_t, 5, root, I).to(device)
    optimization = optim.Adam(model.parameters(), lr=learningrate_t)
    scheduler = optim.lr_scheduler.StepLR(optimization, step_size=1000, gamma=0.8)

    for epoch in range(epochs_t):
        I = data.getall()
        I = I.type(t.float)
        beta = beta0_get(Dt, I)           # beta0 init
        model.train()
        (beta, S) = model(beta, I, Dt)

        loss0 = loss(I, t.matmul(Dt, beta))
        loss1 = loss(beta, t.zeros_like(beta))
        loss2 = loss(beta, S)
        loss_all = loss0 + 0.01*loss1 + 0.01*loss2

        optimization.zero_grad()
        loss_all.backward()
        optimization.step()
        if loss_all <= best_loss:
            best_loss = loss_all
            scio.savemat("./dataout/beta.mat", {'beta': beta.detach().cpu().numpy()})
            t.save(model.state_dict(), './dataout/target.pkl')
        print('epoch:'+str(epoch), '--best_loss:'+str(best_loss), '--loss:'+str(loss_all.data),
              '--loss0:'+str(loss0.data), '--loss2:'+str(loss2.data))
        scheduler.step()
        beta_mat[epoch, :, :] = beta.squeeze(-1).detach().cpu()
        S_mat[epoch, :, :] = S.squeeze(-1).detach().cpu()

    beta_mat = np.array(beta_mat)  # list->numpy
    scio.savemat("./dataout/beta_mat.mat", {'beta_mat': beta_mat})
    scio.savemat("./dataout/S_t_mat.mat", {'S_t_mat': S_mat})

if __name__ == "__main__":
    root = "./dataset/base"
    train_background(root)
    train_target(root)

