#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


class AutoEncoder(nn.Module):
    def __init__(self, args, input_dim):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.latent_dim = args.latent_dim
        self.hidden_dims = args.hidden_dims
        self.hidden_dims.append(self.latent_dim)
        self.dims_list = (
            args.hidden_dims + args.hidden_dims[:-1][::-1]
        )  # mirrored structure
        self.n_layers = len(self.dims_list)
        self.n_clusters = args.n_clusters

        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == args.latent_dim

        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        "linear0": nn.Linear(self.input_dim, hidden_dim),
                        "activation0": nn.ReLU(),
                    }
                )
            else:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(
                            self.hidden_dims[idx - 1], hidden_dim
                        ),
                        "activation{}".format(idx): nn.ReLU(),
                        "bn{}".format(idx): nn.BatchNorm1d(self.hidden_dims[idx]),
                    }
                )
        self.encoder = nn.Sequential(layers)

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(hidden_dim, self.output_dim),
                    }
                )
            else:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(
                            hidden_dim, tmp_hidden_dims[idx + 1]
                        ),
                        "activation{}".format(idx): nn.ReLU(),
                        "bn{}".format(idx): nn.BatchNorm1d(tmp_hidden_dims[idx + 1]),
                    }
                )
        self.decoder = nn.Sequential(layers)

    def __repr__(self):
        repr_str = "[Structure]: {}-".format(self.input_dim)
        for idx, dim in enumerate(self.dims_list):
            repr_str += "{}-".format(dim)
        repr_str += str(self.output_dim) + "\n"
        repr_str += "[n_layers]: {}".format(self.n_layers) + "\n"
        repr_str += "[n_clusters]: {}".format(self.n_clusters) + "\n"
        repr_str += "[input_dims]: {}".format(self.input_dim)
        return repr_str

    def __str__(self):
        return self.__repr__()

    def forward(self, X, latent=False):
        output = self.encoder(X)
        if latent:
            return output
        return self.decoder(output)

    def decode(self, latent_X):
        return self.decoder(latent_X)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1).float()


class UnFlatten(torch.nn.Module):

    def __init__(self, channel, width) -> None:
        super().__init__()
        self.channel = channel
        self.width = width

    def forward(self, x):
        return x.reshape(-1, self.channel, self.width, self.width)


class ConvAutoEncoder(nn.Module):
    def __init__(self, args, input_dim):
        super(ConvAutoEncoder, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = self.input_dim
        self.latent_dim = args.latent_dim

        # encoder #
        self.encoder_conv = nn.Sequential(
            UnFlatten(channel=1, width=16),                       # [batch, 1, 16, 16]
            nn.Conv2d(1, 32, 5, stride=1),                       # [batch, 32, 12, 12]
            nn.BatchNorm2d(32),                                  # [batch, 32, 12, 12]
            nn.ReLU(),

        )
        self.encoder_maxPool = nn.MaxPool2d(2, stride=2, return_indices=True)  # [batch, 32, 6, 6]
        self.encoder_linear = nn.Sequential(
            Flatten(),                                           # [batch, 1152]
            nn.Linear(32 * 6 * 6, self.latent_dim)
        )

        # decoder #
        self.decoder_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 32 * 6 * 6),
            UnFlatten(channel=32, width=6),
        )                                                       # [batch, 32, 6, 6]
        self.decoder_maxPool = nn.MaxUnpool2d(2, stride=2)      # [batch, 32, 12, 12]
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, 5, stride=1),              # [batch, 1, 16, 16]
            Flatten()
        )

    def forward(self, X, latent=False):
        output = self.encode(X)
        if latent:
            return output
        return self.decode(output)

    def encode(self, X):
        out = self.encoder_conv(X)
        out, self.ind = self.encoder_maxPool(out)
        return self.encoder_linear(out)

    def decode(self, X):
        out = self.decoder_linear(X)
        out = self.decoder_maxPool(out, self.ind)
        return self.decoder_conv(out)



# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self, args, input_dim):
        super(ConvVAE, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = self.input_dim
        self.latent_dim = args.latent_dim
        self.channels = args.n_channels
        self.device = args.device
        self.batch_size = args.batch_size
        self.BCELoss = nn.BCELoss(reduction='sum')
        kernel_size = 4 # (4, 4) kernel
        init_channels = 8 # initial number of filters
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=self.channels, out_channels=init_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size,
            stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, self.latent_dim)

        self.encoder = self.encode # encoder network

        self.fc_log_var = nn.Linear(128, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, 64)
        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size,
            stride=1, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=self.channels, kernel_size=kernel_size,
            stride=2, padding=1
        )

        self.decoder = self.decode # decoder network
        # self.encoder_for_loss = self.encode_for_loss

    def encode(self,x): #, return_mu_var=False
        # if x is flattened, reshape it:
        reflat_the_output = False
        if self.args.is_image and len(x.shape) == 2:
            reflat_the_output = True
            full_dim = self.args.input_dim # [B, C, H, W], or [B, H, W, C]
            x = x.view(-1, full_dim[1],full_dim[2],full_dim[3])
            # if full_dim[1]==full_dim[2], and H=W, then it's [B, H, W, C], so permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2) if full_dim[1]==full_dim[2] else x
        # encoding
        x.to(self.device)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        # log_var = self.fc_log_var(hidden)
        # # get the latent vector through reparameterization
        # z = self.reparameterize(mu, log_var)
        ## if return_mu_var:
        ##     return z, mu, log_var
        #return z
        return mu

    
    # def encode_for_loss(self, x):
    #     return self.encode(x, return_mu_var=True)
    
    def decode(self, z):
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)

        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        reconstruction_flat = reconstruction.view(reconstruction.shape[0], -1)
        return reconstruction_flat

        #self.decoder = self.decode()

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample


    def forward(self, x, latent=False): #, return_mu_var=False
        # if return_mu_var:
        #     output, mu, log_var = self.encoder(x, return_mu_var)
        #     return self.decoder(output), mu, log_var
        #else:
        output = self.encoder(x)
        if latent:
            return output
        return self.decoder(output)

    def loss_function(self, recon_x, x, mu, log_var):
        # how well do input x and output recon_x agree?
        BCE = self.BCELoss(recon_x, x)

        # KLD is Kullback–Leibler divergence -- how much does one learned
        # distribution deviate from another, in this specific case the
        # learned distribution from the unit Gaussian

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # note the negative D_{KL} in appendix B of the paper
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= self.batch_size * x.size()[-1]

        combined_loss = BCE + KLD

        return combined_loss
