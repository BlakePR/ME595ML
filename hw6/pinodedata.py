import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import torchdiffeq
from tqdm import tqdm
import time

from torch.utils.data import DataLoader

from pinode import autoencoder, LatentODE, data_loss, phys_loss, TrajectoriesDataset, CollocationDataset, phys_loss_generator_cuda, phys_loss_generator, AltAutoencoder


# np.random.seed(42)
def generatetrajectories(ntraj, tsteps, A, trainflag):
    nx, nz = A.shape
    nt = len(tsteps)

    if trainflag:
        z1 = np.random.uniform(low=-1.5, high=0.5, size=ntraj)
        z2 = np.random.uniform(low=-1, high=1, size=ntraj)
    else:
        z1 = np.random.uniform(low=-1.5, high=1.5, size=ntraj)
        z2 = np.random.uniform(low=-1, high=1, size=ntraj)
    Z0 = np.column_stack((z1, z2))  # ntraj x nz

    Z = np.zeros((ntraj, nt, nz))

    def zode(t, z):
        return [z[1], z[0] - z[0] ** 3]

    for i in range(ntraj):
        sol = solve_ivp(zode, (tsteps[0], tsteps[-1]), Z0[i, :], t_eval=tsteps)
        Z[i, :, :] = sol.y.T

    # map to high dimensional space
    X = np.zeros((ntraj, nt, nx))
    for i in range(nt):
        X[:, i, :] = Z[:, i, :] ** 3 @ A.T

    return X


def getdata(ntrain, ntest, ncol, t_train, t_test):
    """
    :param ntrain: number of training points
    :param ntest: number of testing points
    :param ncol: number of collocation points
    :param t_train: time discretization for training data
    :param t_test: time discretization for testing data
    :return: Xtrain, Xtest, Xcol, fcol, A
    """
    nz = 2
    nx = 128

    A = np.random.normal(size=(nx, nz))

    Xtrain = generatetrajectories(ntrain, t_train, A, trainflag=True)
    Xtest = generatetrajectories(ntest, t_test, A, trainflag=False)

    # collocation points
    z1 = np.random.uniform(low=0.5, high=1.5, size=ncol)
    z2 = np.random.uniform(low=-1, high=1, size=ncol)
    Zcol = np.column_stack((z1, z2))  # ncol x nz
    hZ = np.column_stack((Zcol[:, 1], Zcol[:, 0] - Zcol[:, 0] ** 3))
    fcol = np.zeros((ncol, nx))
    for i in range(ncol):
        fcol[i, :] = hZ[[i], :] @ (3 * A * Zcol[i, :].T ** 2).T
    Xcol = Zcol**3 @ A.T

    return Xtrain, Xtest, Xcol, fcol, A


def true_encoder(X, A):  # X is npts * nt * nx
    Z3 = X @ np.linalg.pinv(A).T  # pinv is nz x nx
    return np.sign(Z3) * np.abs(Z3) ** (1 / 3)

def plot_z(X, z, A):
    x = X.cpu()
    znew = z.cpu()
    true_z = true_encoder(x, A)
    plt.figure()
    plt.title("True Z (black) vs. Predicted Z (blue)")
    for i in range(0, ntest):
        plt.plot(true_z[i, 0, 0], true_z[i, 0, 1], "ko")
        plt.plot(true_z[i, :, 0], true_z[i, :, 1], "k")
        plt.plot(znew[i, 0, 0], znew[i, 0, 1], "bo")
        plt.plot(znew[i, :, 0], znew[i, :, 1], "b")
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1, 1])

    plt.show()

if __name__ == "__main__":
    # discretization in time for training and test data.  These don't need to be changed.
    nt_train = 11
    nt_test = 21
    t_train = np.linspace(0.0, 1.0, nt_train)
    t_test = np.linspace(0.0, 1.0, nt_test)

    # number of training pts, testing pts, and collocation pts.
    # You will need more training pts and collocation pts eventually (testing pts can remain as is).
    ntrain = 1000
    ntest = 100
    ncol = 10000
    np.random.seed(1)
    Xtrain, Xtest, Xcol, fcol, Amap = getdata(ntrain, ntest, ncol, t_train, t_test)
    np.random.seed(int(time.time()))
    # Xtrain is ntrain x nt_train x nx; reminder nx is 128
    # Xtest is ntest x nt_test x nx
    # Xcol is ncol x nx
    # fcol is ncol x nx and represents f(Xcol)
    # Amap is only needed for final plot (see function below)

    # training loop:
    latentdim = 2
    nx = 128
    # autoencoder = autoencoder(input_dim=nx, latent_dim=latentdim,activation=torch.nn.SiLU).double()
    # latentode = LatentODE(latent_dim=latentdim, hidden_dim=128, num_hidden=1, activation=torch.nn.SiLU).double()
    activ = torch.nn.LeakyReLU
    autoencoder = AltAutoencoder(input_dim=nx, latent_dim=latentdim,activation=activ).double().double()
    latentode = LatentODE(latent_dim=latentdim, hidden_dim=128, num_hidden=2, activation=activ).double()

    print(autoencoder)
    print(latentode)

    load_prev = False
    if load_prev:
        autoencoder.load_state_dict(torch.load("hw6\\autoencoderbest.pth"))
        latentode.load_state_dict(torch.load("hw6\\latentodebest.pth"))
        # autoencoder.load_state_dict(torch.load("hw6\\autoencoder.pth"))
        # latentode.load_state_dict(torch.load("hw6\\latentode.pth"))
        
    #datarecon, datapred, physpred, physrecon
    # data_loss = data_loss(alpha1=5, alpha2=2)
    # phys_loss = phys_loss(alpha3=45, alpha4=55)
    data_loss = data_loss(alpha1=1, alpha2=1)
    phys_loss = phys_loss(alpha3=1, alpha4=1)


    # Epochs = 3000
    Epochs = 600
    testevery = 1
    lr = 1e-3
    # guessnbatches = int(ntrain / 128)
    # batch_size_train = int(ntrain / guessnbatches)
    # batch_size_col = int(ncol / guessnbatches)
    # print("Batch size train: ", batch_size_train)
    # print("Batch size col: ", batch_size_col)
    # print("Number of batches: ", guessnbatches)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # traindataset = TrajectoriesDataset(Xtrain)
    # trainloader = DataLoader(traindataset, batch_size=batch_size_train, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,shuffle=True)
    # coldataset = CollocationDataset(Xcol, fcol)
    # colloader = DataLoader(coldataset, batch_size=batch_size_col, num_workers=4, pin_memory=True,drop_last=True, persistent_workers=True,shuffle=True)

    
    optimizer = torch.optim.Adam(
        list(autoencoder.parameters()) + list(latentode.parameters()), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.7, patience=8, cooldown=5, min_lr=1e-8)

    t_train = torch.tensor(t_train, dtype=torch.float64)
    fcol = torch.tensor(fcol, dtype=torch.float64)
    # tq = tqdm(total = Epochs*len(trainloader))
    tq = tqdm(total = Epochs)
    x = torch.tensor(Xtrain, dtype=torch.float64, requires_grad=True)
    # x0 = x[:, 0, :]
    xcol = torch.tensor(Xcol, dtype=torch.float64, requires_grad=True)
    # tensorobjs = [x, xcol, fcol, autoencoder, latentode]
    x = x.cuda()
    xcol = xcol.cuda()
    fcol = fcol.cuda()
    autoencoder.cuda()
    latentode.cuda()
    t_train = t_train.cuda()
    losses = []
    fourlosses = []
    testloss = []
    besterror = 1e10
    lossnames = ["data reconstruction","data prediction","collocation prediction","collocation reconstruction"]

    Xtests = torch.tensor(Xtest, dtype=torch.float64)
    Xtests = Xtests.cuda()
    for epoch in range(Epochs):
        # for (xbatch), (xcolbatch, fcolbatch) in zip(trainloader, colloader):
        #     xbatch, xcolbatch, fcolbatch = xbatch.cuda(), xcolbatch.cuda(), fcolbatch.cuda()
        #     xcolbatch.requires_grad = True
        optimizer.zero_grad()
        # xn, z = autoencoder(xbatch)
        xn, z = autoencoder(x)
        z_pred = torchdiffeq.odeint(latentode, z[:,0,:], t_train, method="rk4",rtol=1e-9,atol=1e-11).permute(1,0,2)
        xn_pred = autoencoder.decoder(z_pred)

        # dataloss1, dataloss2 = data_loss(xbatch, xn, xbatch, xn_pred)
        dataloss1, dataloss2 = data_loss(x, xn, x, xn_pred)
        # term1, term2 = pinode.phys_loss_generator(autoencoder, latentode, xcol, fcol, latentdim, ncol, nx)
        term1, term2, xcol_recon = phys_loss_generator_cuda(autoencoder, latentode, xcol, fcol, latentdim, ncol, nx)
        # term1, term2, xcol_recon = phys_loss_generator_cuda(autoencoder, latentode, xcolbatch, fcolbatch, latentdim, batch_size_col, nx)

        # physloss1, physloss2 = phys_loss(xcolbatch, xcol_recon, term1, term2)
        physloss1, physloss2 = phys_loss(xcol, xcol_recon, term1, term2)
        loss = dataloss1 + dataloss2 + physloss1 + physloss2
        fourlosses.append((dataloss1.item(), dataloss2.item(), physloss1.item(), physloss2.item()))
        iterloss = loss.item()
        losses.append(iterloss)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 12)
        # torch.nn.utils.clip_grad_norm_(latentode.parameters(), 12)
        optimizer.step()
        scheduler.step(iterloss)
        torch.cuda.empty_cache()
        
        autoencoder.eval()
        latentode.eval()
        _, ztest = autoencoder(Xtests)
        ttest = torch.tensor(t_test, dtype=torch.float64)#.flatten()
        ztestpred = torchdiffeq.odeint(latentode, ztest[:,0,:], ttest,method="rk4",rtol=1e-9,atol=1e-11).permute(1,0,2)
        Xtesthat = autoencoder.decoder(ztestpred).cpu().detach().numpy()
        autoencoder.train()
        latentode.train()
        error = np.mean((Xtest - Xtesthat) ** 2)
        testloss.append(error)
        tq.set_postfix({"Loss": iterloss, "LR": optimizer.param_groups[0]['lr'], "Test Error": error})
        tq.update(1)
        if error < besterror:
            besterror = error
            torch.save(autoencoder.state_dict(), "hw6\\autoencoderbest.pth")
            torch.save(latentode.state_dict(), "hw6\\latentodebest.pth")

        # plot_z(x.detach(), z.detach(), Amap)
        # if epoch % (Epochs//3 + 5) == 0 and epoch != 0:
        #     torch.cuda.empty_cache()
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr/100
        
    torch.save(autoencoder.state_dict(), "hw6\\autoencoder.pth")
    torch.save(latentode.state_dict(), "hw6\\latentode.pth")
    tq.close()
    autoencoder.cpu()
    latentode.cpu()
    autoencoder.eval()
    latentode.eval()
    # autoencoder.load_state_dict(torch.load("hw6\\autoencoderbest.pth"))
    # latentode.load_state_dict(torch.load("hw6\\latentodebest.pth"))
    _, ztest = autoencoder(torch.tensor(Xtest, dtype=torch.float64))
    ttest = torch.tensor(t_test, dtype=torch.float64).flatten()
    ztestpred = torchdiffeq.odeint(latentode, ztest[:,0,:],ttest,method="rk4",rtol=1e-9,atol=1e-11).permute(1,0,2)
    Xhat = autoencoder.decoder(ztestpred).detach().numpy()
    # once you have a prediction for Xhat(t) (ntest x nt_test x nx)
    # this will use this specific projection to Z, to create a plot
    # like the bottom right corner of Fig 3
    Zhat = true_encoder(Xhat, Amap)
    Zhat2 = true_encoder(Xtest, Amap)
    error = np.mean((Xtest - Xhat) ** 2)
    plt.figure()
    plt.plot(losses, label = "Total Loss")
    plt.plot(testloss, label = "Test Error")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    print("Error: ", error)

    fourlosses = np.array(fourlosses)
    plt.figure()
    for i in range(4):
        plt.plot(fourlosses[:,i], label = lossnames[i])
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()

    plt.figure()
    plt.title("Predicted X encoded with true A")
    for i in range(0, ntest):

        plt.plot(Zhat[i, 0, 0], Zhat[i, 0, 1], "bo")
        plt.plot(Zhat[i, :, 0], Zhat[i, :, 1], "b")
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1, 1])

    plt.show()

    plt.figure()
    plt.title("Predicted X encoded with true A vs True X encoded with true A")
    for i in range(0, ntest):

        plt.plot(Zhat[i, 0, 0], Zhat[i, 0, 1], "bo")
        plt.plot(Zhat[i, :, 0], Zhat[i, :, 1], "b")
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1, 1])

        plt.plot(Zhat2[i, 0, 0], Zhat2[i, 0, 1], "ko")
        plt.plot(Zhat2[i, :, 0], Zhat2[i, :, 1], "k")
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1, 1])

    plt.show()

    # plt.figure()
    # plt.title("True X encoded with true A")
    # for i in range(0, ntest):

    #     plt.plot(Zhat2[i, 0, 0], Zhat2[i, 0, 1], "ko")
    #     plt.plot(Zhat2[i, :, 0], Zhat2[i, :, 1], "k")
    #     plt.xlim([-1.5, 1.5])
    #     plt.ylim([-1, 1])

    # plt.show()
    # ztest = ztest.detach().numpy()
    # plt.figure()
    # plt.title("predicted Z")
    # for i in range(0, ntest):

    #     plt.plot(ztest[i, 0, 0], ztest[i, 0, 1], "ko")
    #     plt.plot(ztest[i, :, 0], ztest[i, :, 1], "k")
    #     plt.xlim([-1.5, 1.5])
    #     plt.ylim([-1, 1])

    # # plt.show()

    # plot_z(torch.tensor(Xtest, dtype=torch.float64), torch.tensor(ztest), Amap)


#     import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import Dataset

# class autoencoder(nn.Module):
#     def __init__(self, input_dim, latent_dim, activation=nn.LeakyReLU):
#         super(autoencoder, self).__init__()
#         assert(latent_dim in [2, 4, 8, 16, 32, 64])
#         num_encoder_layers = 7 - int(np.log2(latent_dim))
#         encoder = []
#         # encoder.append(nn.Linear(input_dim, input_dim))
#         # encoder.append(activation())
#         curr_dim = input_dim
#         while curr_dim > latent_dim:
#             encoder.append(nn.Linear(curr_dim, curr_dim//4))
#             encoder.append(activation())
#             curr_dim = max(curr_dim // 4, latent_dim)
#         encoder.pop()
#         # encoder.append(nn.Linear(curr_dim, latent_dim))
#         self.encoder = nn.Sequential(*encoder)
        
#         decoder = []
#         # decoder.append(nn.Linear(latent_dim, latent_dim))
#         # decoder.append(activation())
#         curr_dim = latent_dim
#         while curr_dim < input_dim:
#             decoder.append(nn.Linear(curr_dim, curr_dim * 4))
#             decoder.append(activation())
#             curr_dim = min(curr_dim * 4, input_dim)
#         # decoder.append(nn.Linear(curr_dim, input_dim))
#         decoder.pop()
#         self.decoder = nn.Sequential(*decoder)

#     def forward(self, x):
#         z = self.encoder(x)
#         new_x = self.decoder(z)
#         return new_x, z

# class AltAutoencoder(nn.Module):
#     def __init__(self, input_dim, latent_dim, activation=nn.LeakyReLU):
#         super(AltAutoencoder, self).__init__()
#         encoder = []
#         encoder.append(nn.Linear(input_dim, 256))
#         encoder.append(activation())
#         encoder.append(nn.Linear(256, 256))
#         encoder.append(activation())
#         encoder.append(nn.Linear(256, 256))
#         encoder.append(activation())
#         encoder.append(nn.Linear(256, latent_dim))
#         self.encoder = nn.Sequential(*encoder)
        
#         decoder = []
#         decoder.append(nn.Linear(latent_dim, 256))
#         decoder.append(activation())
#         decoder.append(nn.Linear(256, 256))
#         decoder.append(activation())
#         decoder.append(nn.Linear(256, 256))
#         decoder.append(activation())
#         decoder.append(nn.Linear(256, input_dim))
#         self.decoder = nn.Sequential(*decoder)

#     def forward(self, x):
#         z = self.encoder(x)
#         new_x = self.decoder(z)
#         return new_x, z
    
# class LatentODE(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, num_hidden = 1, activation=nn.LeakyReLU):
#         super(LatentODE, self).__init__()

#         layers = []
#         layers.append(nn.Linear(latent_dim, hidden_dim))
#         layers.append(activation())
#         for i in range(num_hidden):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(activation())
#         layers.append(nn.Linear(hidden_dim, latent_dim))
#         self.net = nn.Sequential(*layers)

#     def forward(self, t, z):
#         return self.net(z)

# class data_loss():
#     def __init__(self, alpha1=1, alpha2=1):
#         self.alpha1 = alpha1
#         self.alpha2 = alpha2
#         self.mse = nn.MSELoss()

#     def __call__(self, x, x_pred, y, y_pred):
#         reconstruction_loss = torch.mean((x - x_pred) ** 2)
#         prediction_loss = torch.mean((y - y_pred) ** 2)
#         return self.alpha1 * reconstruction_loss, self.alpha2 * prediction_loss

# class phys_loss():
#     def __init__(self, alpha3=1, alpha4=1):
#         # :param alpha3: weight for latent gradient loss
#         # :param alpha4: weight for collocation reconstruction loss
#         self.alpha3 = alpha3
#         self.alpha4 = alpha4
#         self.mse = nn.MSELoss()

#     def __call__(self, x, x_pred, dphidx_f, hphix):
#         latent_gradient_loss = torch.mean((dphidx_f - hphix) ** 2)
#         collocation_reconstruction_loss = torch.mean((x - x_pred) ** 2)
#         return self.alpha3 * latent_gradient_loss, self.alpha4 * collocation_reconstruction_loss

# def phys_loss_generator(encoder, latentode, x_col, f_x, nz, ncol,nx):
#     # dphi/dx * f = h(phi(x))
#     xcol_recon, z_col = encoder(x_col) # xcol is ncol x nx, z_col is ncol x nz
#     zdot_col = latentode(0,z_col) #h(phi(x)) is ncol x nz
#     dzdx = torch.zeros(ncol, nz, nx).double()
#     # dzdx = torch.zeros(ncol, nz, nx).cuda()
#     for i in range(nz):
#         dzdx[:,i,:] = torch.autograd.grad(zdot_col[:,i], x_col, grad_outputs=torch.ones_like(zdot_col[:,i]), create_graph=True)[0]
#     return torch.bmm(dzdx,f_x.unsqueeze(2)).squeeze(), zdot_col, xcol_recon

# def phys_loss_generator_cuda(encoder, latentode, x_col, f_x, nz, ncol,nx):
#     # dphi/dx * f = h(phi(x))
#     xcol_recon, z_col = encoder(x_col) # xcol is ncol x nx, z_col is ncol x nz
#     zdot_col = latentode(0,z_col) #h(phi(x)) is ncol x nz
#     # dzdx = torch.zeros(ncol, nz, nx)
#     dzdx = torch.zeros(ncol, nz, nx).cuda().double()
#     for i in range(nz):
#         dzdx[:,i,:] = torch.autograd.grad(z_col[:,i], x_col, grad_outputs=torch.ones_like(zdot_col[:,i]), create_graph=True)[0]
#     return torch.bmm(dzdx,f_x.unsqueeze(2)).squeeze(), zdot_col, xcol_recon

# class TrajectoriesDataset(Dataset):
#     def __init__(self, X):
#         self.X = torch.tensor(X, dtype=torch.float64) # ntraj x nt x nx

#     def __len__(self):
#         return self.X.shape[0] # ntraj

#     def __getitem__(self, idx):
#         return self.X[idx] # nt x nx

# class CollocationDataset(Dataset):
#     def __init__(self, xcol, fcol):
#         self.xcol = torch.tensor(xcol, dtype=torch.float64)
#         self.fcol = torch.tensor(fcol, dtype=torch.float64)

#     def __len__(self):
#         return self.xcol.shape[0]

#     def __getitem__(self, idx):
#         return self.xcol[idx], self.fcol[idx]

