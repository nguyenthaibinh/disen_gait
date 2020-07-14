import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Encoder(nn.Module):
    def __init__(self, in_size=(1, 64, 64), hid_channels=32, hidden_dim=256, id_dim=96, cov_dim=32,
                 kernel_size=4):
        super(Encoder, self).__init__()
        nc = in_size[0]
        self.id_dim = id_dim
        self.cov_dim = cov_dim
        self.latent_dim = id_dim + cov_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(nc, hid_channels, 4, 2, 1, bias=True),          # B,  32, 64, 64
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(hid_channels, hid_channels, 4, 2, 1, bias=True),          # B,  32, 32, 32
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(hid_channels, hid_channels, 4, 2, 1, bias=True),          # B,  32, 16, 16
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(hid_channels, hid_channels, 4, 2, 1, bias=True),          # B,  32,  8,  8
            nn.ReLU(True)
        )

        self.fc_id = nn.Sequential(
            nn.Linear(32 * 4 * 4, hidden_dim, bias=True),  # B, 256
            nn.ReLU(True),
            nn.Linear(hidden_dim, self.id_dim, bias=True)
        )

        self.fc_cov_mu = nn.Sequential(
            nn.Linear(32 * 4 * 4, hidden_dim, bias=True),  # B, 256
            nn.ReLU(True),
            nn.Linear(hidden_dim, self.cov_dim, bias=True)
        )

        self.fc_cov_logvar = nn.Sequential(
            nn.Linear(32 * 4 * 4, hidden_dim, bias=True),  # B, 256
            nn.ReLU(True),
            nn.Linear(hidden_dim, self.cov_dim, bias=True)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        h = self.conv_layers(x)
        h = h.view((batch_size, -1))
        z_id = self.fc_id(h)
        cov_mu = self.fc_cov_mu(h)
        cov_logvar = self.fc_cov_logvar(h)

        return z_id, cov_mu, cov_logvar


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

        self.id_dim = id_dim
        self.cov_dim = cov_dim
        self.latent_dim = self.id_dim + self.cov_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim, bias=True),  # B, 256
            nn.ReLU(True),
            nn.Linear(hidden_dim, 32 * 4 * 4, bias=True),  # B, 512
            nn.ReLU(True)
        )

        self.t_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(hid_channels, hid_channels, 4, 2, 1, bias=True),  # B,  32,  8,  8
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(hid_channels, hid_channels, 4, 2, 1, bias=True),  # B,  32, 16, 16
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(hid_channels, hid_channels, 4, 2, 1, bias=True),  # B,  32, 32, 32
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(hid_channels, 1, 4, 2, 1, bias=True),  # B,  nc, 64, 64
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


class Discriminator(nn.Module):
    def __init__(self, hidden_dim=512):
        super(Discriminator, self).__init__()

        self.conv_model = nn.Sequential(OrderedDict([
            ('convolution_1',
             nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True)),
            ('convolution_1_in', nn.InstanceNorm2d(num_features=32, track_running_stats=True)),
            ('LeakyReLU_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('convolution_2',
             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True)),
            ('convolution_2_in', nn.InstanceNorm2d(num_features=64, track_running_stats=True)),
            ('LeakyReLU_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('convolution_3',
             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True)),
            ('convolution_3_in', nn.InstanceNorm2d(num_features=128, track_running_stats=True)),
            ('LeakyReLU_3', nn.LeakyReLU(negative_slope=0.2, inplace=True))
        ]))

        self.fully_connected_model = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(in_features=6272, out_features=hidden_dim, bias=True)),
            ('LeakyReLU', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
            ('fc_2', nn.Linear(in_features=hidden_dim, out_features=1, bias=True)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, image_1, image_2):
        x = th.cat((image_1, image_2), dim=1)
        x = self.conv_model(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fully_connected_model(x)

        return x


class Classifier(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Classifier, self).__init__()

        self.fc_model = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(in_features=z_dim, out_features=256, bias=True)),
            ('fc_1_bn', nn.BatchNorm1d(num_features=256)),
            ('LeakyRelu_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_2', nn.Linear(in_features=256, out_features=256, bias=True)),
            ('fc_2_bn', nn.BatchNorm1d(num_features=256)),
            ('LeakyRelu_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_3', nn.Linear(in_features=256, out_features=num_classes, bias=True))
        ]))

    def forward(self, z):
        x = self.fc_model(z)

        return x
