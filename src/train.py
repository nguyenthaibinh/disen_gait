import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import TripletDataset
from accuracy import knn_evaluate

from nets.disentanged_autoencoder import DlrAutoEncoder
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

class DlrAutoEncTrainer():
    def __init__(self, in_size=(1, 64, 64), lr=0.001, beta=250, beta1=0.5, beta2=0.999,
                 margin=2.0, device='gpu', log_dir='./tensor_log', multi_gpu=False,
                 p_dropout=0.5, num_workers=8, checkpoint_freq=100, neg_ratio=0.5,
                 checkpoint_dir='./checkpoint', feature_dir='./feature_vectors'):
        super(DlrAutoEncTrainer, self).__init__()
        self.device = device
        self.g_net = DlrAutoEncoder()
        self.d_net = Discriminator()

        if multi_gpu == 1:
            if th.cuda.device_count() > 1:
                print("Let's use", th.cuda.device_count(), "GPUs!")
                self.net = nn.DataParallel(self.net)
                self.d_net = nn.DataParallel(self.d_net)
        else:
            self.g_net = self.g_net.to(device, dtype=th.float)
            self.d_net = self.d_net.to(device, dtype=th.float)

        self.g_net.apply(weights_init)
        self.d_net.apply(weights_init)

        # Setup Adam optimizers for both G and D
        self.g_optim = optim.Adam(self.g_net.parameters(), lr=lr)
        self.d_optim = optim.Adam(self.d_net.parameters(), lr=lr)

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

    def zero_grad(self):
        self.g_optim.zero_grad()
        self.d_optim.zero_grad()

    def compute_loss(self, x1, x2, x1_hat, x2_hat, x1_swap, x2_swap, z1_id, z2_id, labels):
        # compute reconstruction loss
        recon_loss = 0
        for i in range(len(x1)):
            recon_loss += (1 - labels[i]) * F.mse_loss(x1[i], x1_swap[i])
            recon_loss += (1 - labels[i]) * F.mse_loss(x2[i], x2_swap[i])
            recon_loss += labels[i] * F.mse_loss(x1[i], x1_hat[i])
            recon_loss += labels[i] * F.mse_loss(x2[i], x2_hat[i])

        contrastive_loss = get_contrastive_loss(z1_id, z2_id, labels, self.margin)

        return recon_loss, contrastive_loss

    def train_epoch(self, data_loader, epoch):
        self.g_net.train()
        self.d_net.train()

        epoch_recon_losses = []
        epoch_triplet_losses = []
        epoch_feature_losses = []

        for i, batch_data in enumerate(data_loader):
            self.zero_grad()

            x1, x2, x3 = batch_data

            x1 = x1.to(self.device, dtype=th.float)
            x2 = x2.to(self.device, dtype=th.float)
            x3 = x3.to(self.device, dtype=th.float)

            x1_hat, x3_hat, x12_hat, x32_hat, z1_id, z1_cov, z2_id, z2_cov, z3_id, z3_cov, \
            z12_id, z12_cov, z32_id, z32_cov = self.g_net(x1, x2, x3)

            # update the discriminator
            ## feed real data
            b_size = len(x2)
            real_label = th.full((b_size,), self.real_label, device=self.device)
            scores = self.d_net(x2)
            err_d_real = F.binary_cross_entropy(scores, real_label)

            ## feed fake data
            b_size = len(x12_hat)
            fake_label = th.full((b_size,), self.fake_label, device=self.device)
            scores = self.d_net(x12_hat.detach())
            err_d_fake = F.binary_cross_entropy(scores, fake_label)
            err_d = err_d_real + err_d_fake
            err_d.backward()
            self.d_optim.step()

            # update the auto_encoder
            # recon loss
            recon_loss = F.mse_loss(x1_hat, x1)  # + F.mse_loss(x3_hat, x3)

            # Update encoder-decoder net
            real_label = th.full((b_size,), self.real_label, device=self.device)
            scores = self.d_net(x12_hat)
            err_d_g_real = F.binary_cross_entropy(scores, real_label)

            # feature_loss
            # feature_loss = F.mse_loss(z12_id, z1_id) + F.mse_loss(z32_id, z3_id)
            feature_loss = F.mse_loss(z12_cov, z2_cov) + F.mse_loss(z32_cov, z2_cov)

            # triplet loss
            triplet_loss = get_triplet_loss(z1_id, z2_id, z3_id, margin=self.margin)

            # loss = recon_loss + feature_loss + triplet_loss
            err_g = recon_loss + err_d_g_real
            err_g.backward()
            self.g_optim.step()

            epoch_recon_losses.append(recon_loss.item())
            epoch_triplet_losses.append(triplet_loss.item())
            epoch_feature_losses.append(feature_loss.item())

        recon_loss = np.array(epoch_recon_losses).mean()
        triplet_loss = np.array(epoch_triplet_losses).mean()
        feature_loss = np.array(epoch_feature_losses).mean()

        return recon_loss, triplet_loss, feature_loss

    def compute_d_loss(self, x_real, x_fake):
        # update discriminator
        ## update with real batch
        b_size = len(x_real)
        real_label = th.full((b_size,), self.real_label, device=self.device)
        scores = self.d_net(x_real)
        err_d_real = F.binary_cross_entropy(scores, real_label)

        ## update with fake batch
        b_size = len(x_fake)
        fake_label = th.full((b_size,), self.fake_label, device=self.device)
        scores = self.d_net(x_fake.detach())
        err_d_fake = F.binary_cross_entropy(scores, fake_label)
        err_d_fake.backward()
        err_d = err_d_real + err_d_fake
        self.d_optim.step()

        # Update encoder-decoder net
        real_label = th.full((b_size,), self.real_label, device=self.device)
        scores = self.d_net(x_fake)
        g_err = F.binary_cross_entropy(scores, real_label)
        g_err.backward()
        self.g_optim.step()
        return 0

    def train(self, train_set, validation_set, num_epochs=100, verbose=True):
        for epoch in range(1, num_epochs + 1):
            data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True,
                                     num_workers=self.num_workers)
            recon_loss, triplet_loss, feature_loss = self.train_epoch(data_loader, epoch)

            self.writer.add_scalars('recon_loss', {'train': recon_loss}, epoch)
            self.writer.add_scalars('contrastive_loss', {'train': triplet_loss}, epoch)
            self.writer.add_scalars('discrimination_loss', {'train': feature_loss}, epoch)

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

            if verbose is True:
                print(f'Epoch {epoch + 0:0005}/{num_epochs}: recon_loss: {recon_loss:.5f}, '
                      f'triplet_loss: {triplet_loss:.5f}, feature_loss: {feature_loss:.5f}.')

    def visualize(self, epoch, dataset, size=32):
        data_loader = DataLoader(dataset=dataset, batch_size=size, shuffle=False,
                                 num_workers=self.num_workers)
        self.g_net.eval()
        self.d_net.eval()
        for i, batch_data in enumerate(data_loader):
            x1, x2, x3 = batch_data
            x1 = x1.to(self.device, dtype=th.float)
            x2 = x2.to(self.device, dtype=th.float)
            x3 = x3.to(self.device, dtype=th.float)

            x1_hat, x3_hat, x12_hat, x32_hat, z1_id, z1_cov, z2_id, z2_cov, z3_id, z3_cov, \
            z12_id, z12_cov, z32_id, z32_cov = self.g_net(x1, x2, x3)

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
