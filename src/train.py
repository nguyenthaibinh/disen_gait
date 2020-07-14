import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import TripletDataset
from metrics import knn_evaluate
from pathlib import Path

from nets.disentanged_autoencoder import DlrAutoEncoder
from nets.encoders import Encoder
from nets.decoders import Decoder
from nets.discriminators import Discriminator

def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()

def get_contrastive_loss(x1, x2, label, margin):
    euclidean_distance = F.pairwise_distance(x1, x2, keepdim=True)
    contrastive_loss = th.mean((1 - label) * th.pow(euclidean_distance, 2) +
                               label * th.pow(th.clamp(margin - euclidean_distance, min=0.0), 2))
    return contrastive_loss

def get_triplet_loss(anchor, positive, negative, margin, size_average=True):
    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean() if size_average else losses.sum()

class DlrAutoEncTrainer(object):
    def __init__(self, in_size=(1, 64, 64), lr=0.001, beta=250, beta1=0.5, beta2=0.999,
                 margin=2.0, device='gpu', log_dir='./tensor_log', multi_gpu=False,
                 p_dropout=0.5, num_workers=8, checkpoint_freq=100, neg_ratio=0.5,
                 checkpoint_dir='./checkpoint', feature_dir='./feature_vectors'):
        super(DlrAutoEncTrainer, self).__init__()
        self.device = device
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()

        if multi_gpu == 1:
            if th.cuda.device_count() > 1:
                print("Let's use", th.cuda.device_count(), "GPUs!")
                self.encoder = nn.DataParallel(self.encoder)
                self.decoder = nn.DataParallel(self.decoder)
                self.discriminator = nn.DataParallel(self.discriminator)
        else:
            self.encoder = self.encoder.to(device, dtype=th.float)
            self.decoder = self.decoder.to(device, dtype=th.float)
            self.discriminator = self.discriminator.to(device, dtype=th.float)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator.apply(weights_init)

        # Setup Adam optimizers for both G and D
        self.ae_optim = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr,
                                   betas=(beta1, beta2))
        self.g_optim = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr,
                                  betas=(beta1, beta2))
        self.d_optim = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

        self.margin = margin
        self.beta = beta
        self.p_dropout = p_dropout
        self.multi_gpu = multi_gpu
        self.feature_dir = feature_dir
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        self.num_workers = num_workers
        self.checkpoint_freq = checkpoint_freq
        self.neg_ratio = neg_ratio

        self.writer = SummaryWriter(self.log_dir)
        self.real_label = 1
        self.fake_label = 0

        self.training = False

    def zero_grad(self):
        self.ae_optim.zero_grad()
        self.g_optim.zero_grad()
        self.d_optim.zero_grad()

    def train_mode(self):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        self.training = True

    def eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        self.training = False

    def train_epoch(self, data_loader, epoch):
        self.train_mode()

        epoch_recon_losses = []
        epoch_triplet_losses = []
        epoch_feature_losses = []
        epoch_z_pos_distances = []
        epoch_z_neg_distances = []

        for i, batch_data in enumerate(data_loader):
            self.zero_grad()

            x1, x2, x3 = batch_data

            x1 = x1.to(self.device, dtype=th.float)
            x2 = x2.to(self.device, dtype=th.float)
            x3 = x3.to(self.device, dtype=th.float)

            # train auto encoder
            z1_id, z1_cov = self.encoder(x1)
            z2_id, z2_cov = self.encoder(x2)
            z3_id, _ = self.encoder(x3)

            x1_hat = self.decoder(z1_id, z1_cov)
            x12_hat = self.decoder(z1_id, z2_cov)

            triplet_loss = get_triplet_loss(z1_id, z2_id, z3_id, self.margin)

            recon_loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x12_hat, x2)

            # z_loss
            zero_cov = np.zeros(shape=z1_cov.shape)
            zero_cov = th.from_numpy(zero_cov).to(self.device, dtype=th.float)
            x10_hat = self.decoder(z1_id, zero_cov)
            z10_id, _ = self.encoder(x10_hat)
            z_loss = F.mse_loss(z1_id, z10_id)

            loss = recon_loss + triplet_loss + z_loss
            loss.backward()
            self.g_optim.step()

            epoch_z_pos_distances.append(F.pairwise_distance(z1_id, z2_id).mean().item())
            epoch_z_neg_distances.append(F.pairwise_distance(z1_id, z3_id).mean().item())

            epoch_recon_losses.append(recon_loss.item())
            epoch_triplet_losses.append(triplet_loss.item())
            epoch_feature_losses.append(z_loss.item())

        recon_loss = np.array(epoch_recon_losses).mean()
        triplet_loss = np.array(epoch_triplet_losses).mean()
        feature_loss = np.array(epoch_feature_losses).mean()
        z_pos_distance = np.array(epoch_z_pos_distances).mean()
        z_neg_distance = np.array(epoch_z_neg_distances).mean()

        return recon_loss, triplet_loss, z_pos_distance, z_neg_distance

    def train(self, train_set, validation_set, num_epochs=100, verbose=True):
        for epoch in range(1, num_epochs + 1):
            data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True,
                                     num_workers=self.num_workers)
            recon_loss, triplet_loss, z_pos_dist, z_neg_dis = self.train_epoch(data_loader, epoch)

            self.writer.add_scalars('recon_loss', {'train': recon_loss}, epoch)
            self.writer.add_scalars('triplet_loss', {'train': triplet_loss}, epoch)
            # self.writer.add_scalars('discrimination_loss', {'train': feature_loss}, epoch)

            """
            val_hit_1, _, _ = self.compute_accuracy(val_probe_data, val_probe_labels,
                                                    val_gallery_data, val_gallery_labels, k=1)
            val_hit_5, _, _ = self.compute_accuracy(val_probe_data, val_probe_labels,
                                                    val_gallery_data, val_gallery_labels, k=5)
            self.writer.add_scalars('val_hit_rate', {'rank@1': val_hit_1, 'rank@5': val_hit_5}, epoch)

            if epoch % self.checkpoint_freq == 0:
                model_name = f'snapshot_epoch_{epoch}.pth'
                self.save_model(model_name)
            """

            self.visualize(epoch, train_set, size=32)

            if epoch % 100 == 0:
                self.save_model(epoch)

            if verbose is True:
                print(f'Epoch {epoch + 0:0005}/{num_epochs}: recon_loss: {recon_loss:.5f}, '
                      f'triplet_loss: {triplet_loss:.5f}, z_pos_dis: {z_pos_dist:.5f}, '
                      f'z_neg_dis: {z_neg_dis:0.5f}.')

    def visualize(self, epoch, dataset, size=32):
        data_loader = DataLoader(dataset=dataset, batch_size=size, shuffle=False,
                                 num_workers=self.num_workers)
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        for i, batch_data in enumerate(data_loader):
            x1, x2, x3 = batch_data
            x1 = x1.to(self.device, dtype=th.float)
            x2 = x2.to(self.device, dtype=th.float)
            x3 = x3.to(self.device, dtype=th.float)

            z1_id, z1_cov = self.encoder(x1)
            z2_id, z2_cov = self.encoder(x2)
            z3_id, z3_cov = self.encoder(x3)

            x1_hat = self.decoder(z1_id, z1_cov)
            x12_hat = self.decoder(z1_id, z2_cov)
            x32_hat = self.decoder(z3_id, z2_cov)
            x3_hat = self.decoder(z3_id, z3_cov)

            self.plot_images(x1, x2, x3, x1_hat, x12_hat, x32_hat, x3_hat, epoch)
            break

    def plot_images(self, x1, x2, x3, x1_hat, x12_hat, x32_hat, x3_hat, epoch):
        x1 = x1.detach().cpu().numpy()
        x2 = x2.detach().cpu().numpy()
        x3 = x3.detach().cpu().numpy()
        x1_hat = x1_hat.detach().cpu().numpy()
        x12_hat = x12_hat.detach().cpu().numpy()
        x32_hat = x32_hat.detach().cpu().numpy()
        x3_hat = x3_hat.detach().cpu().numpy()
        n_len = len(x1)
        img_batch = []
        for i in range(n_len):
            img_batch.append(th.from_numpy(x1[i, :, :, :]))
            img_batch.append(th.from_numpy(x2[i, :, :, :]))
            img_batch.append(th.from_numpy(x3[i, :, :, :]))
            img_batch.append(th.from_numpy(x1_hat[i, :, :, :]))
            img_batch.append(th.from_numpy(x12_hat[i, :, :, :]))
            img_batch.append(th.from_numpy(x32_hat[i, :, :, :]))
            img_batch.append(th.from_numpy(x3_hat[i, :, :, :]))
        image_grid = tv.utils.make_grid(img_batch, nrow=7)
        self.writer.add_image(f'Two recon_losses',
                              image_grid, global_step=epoch)

    def compute_accuracy(self, probe_data, probe_labels, gallery_data, gallery_labels, k=5, remove_first=False):
        gallery_z_id = self.inference(gallery_data)
        probe_z_id = self.inference(probe_data)

        hit_rate, pre, recall = knn_evaluate(probe_z_id, probe_labels, gallery_z_id, gallery_labels,
                                             k=k, remove_first=remove_first)
        return hit_rate, pre, recall

    def inference(self, data):
        self.net.eval()
        data = th.from_numpy(data).to(self.device, dtype=th.float)
        try:
            z_id, _ = self.net.encode(data)
            z_id = z_id.detach().cpu().numpy()
        except Exception as e:
            print(e)
            z_id = None
        return z_id

    def save_model(self, epoch):
        encoder_path = Path(self.checkpoint_dir, f"encoder_epoch_{epoch:05}.pth")
        decoder_path = Path(self.checkpoint_dir, f"decoder_epoch_{epoch:05}.pth")
        discriminator_path = Path(self.checkpoint_dir, f"discriminator_epoch_{epoch:05}.pth")
        if encoder_path.exists():
            encoder_path.unlink()
        if decoder_path.exists():
            decoder_path.unlink()
        if discriminator_path.exists():
            discriminator_path.unlink()

        th.save(self.encoder.state_dict(), str(encoder_path))
        th.save(self.decoder.state_dict(), str(decoder_path))
        th.save(self.discriminator.state_dict(), str(discriminator_path))

