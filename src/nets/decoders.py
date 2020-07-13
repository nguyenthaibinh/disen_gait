"""
Module containing the decoders.
"""
import numpy as np

import torch as th
from torch import nn


# ALL decoders should be called Decoder<Model>
def get_decoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Decoder{}".format(model_type))


class Decoder(nn.Module):
    def __init__(self, out_size=(1, 64, 64), id_dim=96, cov_dim=32, hid_channels=32, hidden_dim=256, kernel_size=4):
        r"""Decoder of the model proposed in [1].
        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_dim : int
            Dimensionality of latent output.
        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(Decoder, self).__init__()

        out_channels = out_size[0]

        self.id_dim = id_dim
        self.cov_dim = cov_dim
        self.latent_dim = self.id_dim + self.cov_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim, bias=True),  # B, 256
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim, bias=True),  # B, 256
            nn.ReLU(True),
            nn.Linear(hidden_dim, 32 * 4 * 4, bias=True),  # B, 512
            nn.ReLU(True)
        )

        self.t_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=True),  # B,  32,  8,  8
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=True),  # B,  32, 16, 16
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=True),  # B,  32, 32, 32
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=True),  # B,  nc, 64, 64
            nn.Sigmoid()
        )

    def forward(self, z_id, z_cov):
        z = th.cat((z_id, z_cov), dim=1)

        # Fully connected layers with ReLu activations
        z = self.fc_layers(z)
        z = z.view((-1, 32, 4, 4))  # B,  32,  4,  4
        # z = z.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        x_hat = self.t_conv_layers(z)
        return x_hat