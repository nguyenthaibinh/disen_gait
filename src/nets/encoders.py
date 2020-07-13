"""
Module containing the encoders.
"""
import numpy as np

import torch as th
from torch import nn


# ALL encoders should be called Enccoder<Model>
def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Encoder{}".format(model_type))


class Encoder(nn.Module):
    def __init__(self, in_size=(1, 64, 64), hid_channels=32, hidden_dim=256, id_dim=96, cov_dim=32,
                 kernel_size=4):
        super(Encoder, self).__init__()
        nc = in_size[0]
        self.id_dim = id_dim
        self.cov_dim = cov_dim
        self.latent_dim = id_dim + cov_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1, bias=True),          # B,  32, 64, 64
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1, bias=True),          # B,  32, 32, 32
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1, bias=True),          # B,  32, 16, 16
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1, bias=True),          # B,  32,  8,  8
            nn.ReLU(True)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 4 * 4, hidden_dim, bias=True),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, hidden_dim, bias=True),         # B, z_dim*2
            nn.ReLU(True),
            nn.Linear(hidden_dim, self.latent_dim, bias=True)
        )

        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        h = self.conv_layers(x)
        h = h.view((batch_size, -1))
        h = self.fc_layers(h)
        z_id = h[:, :self.id_dim]
        z_cov = h[:, self.id_dim:]

        return z_id, z_cov