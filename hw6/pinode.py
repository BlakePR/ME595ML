import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, activation=nn.LeakyReLU):
        super(autoencoder, self).__init__()
        assert(latent_dim in [2, 4, 8, 16, 32, 64])
        num_encoder_layers = 7 - int(np.log2(latent_dim))
        encoder = []
        # encoder.append(nn.Linear(input_dim, input_dim))
        # encoder.append(activation())
        curr_dim = input_dim
        while curr_dim > latent_dim:
            encoder.append(nn.Linear(curr_dim, curr_dim//4))
            encoder.append(activation())
            curr_dim = max(curr_dim // 4, latent_dim)
        encoder.pop()
        # encoder.append(nn.Linear(curr_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)
        
        decoder = []
        # decoder.append(nn.Linear(latent_dim, latent_dim))
        # decoder.append(activation())
        curr_dim = latent_dim
        while curr_dim < input_dim:
            decoder.append(nn.Linear(curr_dim, curr_dim * 4))
            decoder.append(activation())
            curr_dim = min(curr_dim * 4, input_dim)
        # decoder.append(nn.Linear(curr_dim, input_dim))
        decoder.pop()
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        z = self.encoder(x)
        new_x = self.decoder(z)
        return new_x, z

class AltAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, activation=nn.LeakyReLU):
        super(AltAutoencoder, self).__init__()
        encoder = []
        encoder.append(nn.Linear(input_dim, 256))
        encoder.append(activation())
        encoder.append(nn.Linear(256, 256))
        encoder.append(activation())
        encoder.append(nn.Linear(256, 256))
        encoder.append(activation())
        encoder.append(nn.Linear(256, latent_dim))
        self.encoder = nn.Sequential(*encoder)
        
        decoder = []
        decoder.append(nn.Linear(latent_dim, 256))
        decoder.append(activation())
        decoder.append(nn.Linear(256, 256))
        decoder.append(activation())
        decoder.append(nn.Linear(256, 256))
        decoder.append(activation())
        decoder.append(nn.Linear(256, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        z = self.encoder(x)
        new_x = self.decoder(z)
        return new_x, z
    
class LatentODE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_hidden = 1, activation=nn.LeakyReLU):
        super(LatentODE, self).__init__()

        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(activation())
        for i in range(num_hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, z):
        return self.net(z)

class data_loss():
    def __init__(self, alpha1=1, alpha2=1):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.mse = nn.MSELoss()

    def __call__(self, x, x_pred, y, y_pred):
        reconstruction_loss = torch.mean((x - x_pred) ** 2)
        prediction_loss = torch.mean((y - y_pred) ** 2)
        return self.alpha1 * reconstruction_loss, self.alpha2 * prediction_loss

class phys_loss():
    def __init__(self, alpha3=1, alpha4=1):
        # :param alpha3: weight for latent gradient loss
        # :param alpha4: weight for collocation reconstruction loss
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        self.mse = nn.MSELoss()

    def __call__(self, x, x_pred, dphidx_f, hphix):
        latent_gradient_loss = torch.mean((dphidx_f - hphix) ** 2)
        collocation_reconstruction_loss = torch.mean((x - x_pred) ** 2)
        return self.alpha3 * latent_gradient_loss, self.alpha4 * collocation_reconstruction_loss

def phys_loss_generator(encoder, latentode, x_col, f_x, nz, ncol,nx):
    # dphi/dx * f = h(phi(x))
    xcol_recon, z_col = encoder(x_col) # xcol is ncol x nx, z_col is ncol x nz
    zdot_col = latentode(0,z_col) #h(phi(x)) is ncol x nz
    dzdx = torch.zeros(ncol, nz, nx).double()
    # dzdx = torch.zeros(ncol, nz, nx).cuda()
    for i in range(nz):
        dzdx[:,i,:] = torch.autograd.grad(zdot_col[:,i], x_col, grad_outputs=torch.ones_like(zdot_col[:,i]), create_graph=True)[0]
    return torch.bmm(dzdx,f_x.unsqueeze(2)).squeeze(), zdot_col, xcol_recon

def phys_loss_generator_cuda(encoder, latentode, x_col, f_x, nz, ncol,nx):
    # dphi/dx * f = h(phi(x))
    xcol_recon, z_col = encoder(x_col) # xcol is ncol x nx, z_col is ncol x nz
    zdot_col = latentode(0,z_col) #h(phi(x)) is ncol x nz
    # dzdx = torch.zeros(ncol, nz, nx)
    dzdx = torch.zeros(ncol, nz, nx).cuda().double()
    for i in range(nz):
        dzdx[:,i,:] = torch.autograd.grad(z_col[:,i], x_col, grad_outputs=torch.ones_like(zdot_col[:,i]), create_graph=True)[0]
    return torch.bmm(dzdx,f_x.unsqueeze(2)).squeeze(), zdot_col, xcol_recon

class TrajectoriesDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float64) # ntraj x nt x nx

    def __len__(self):
        return self.X.shape[0] # ntraj

    def __getitem__(self, idx):
        return self.X[idx] # nt x nx

class CollocationDataset(Dataset):
    def __init__(self, xcol, fcol):
        self.xcol = torch.tensor(xcol, dtype=torch.float64)
        self.fcol = torch.tensor(fcol, dtype=torch.float64)

    def __len__(self):
        return self.xcol.shape[0]

    def __getitem__(self, idx):
        return self.xcol[idx], self.fcol[idx]

