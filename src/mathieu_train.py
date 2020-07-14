import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from metrics import knn_evaluate
from nets.mathieu_net import Encoder, Decoder, Discriminator
from itertools import cycle

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

def kl_divergence(mu, logvar):
    kl_div = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_div

def reparameterize(training, mu, logvar):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:
        return mu

class MathieuTrainer():
    def __init__(self, in_size=(1, 64, 64), lr=0.001, beta=250, beta1=0.5, beta2=0.999,
                 margin=2.0, device='gpu', log_dir='./tensor_log', multi_gpu=False,
                 p_dropout=0.5, num_workers=8, checkpoint_freq=100, neg_ratio=0.5,
                 checkpoint_dir='./checkpoint', feature_dir='./feature_vectors'):
        super(MathieuTrainer, self).__init__()
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
        self.autoenc_optim = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                        lr=lr, betas=(beta1, beta2))
        self.generator_optim = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                          lr=lr, betas=(beta1, beta2))
        self.discriminator_optim = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

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
        self.autoenc_optim.zero_grad()
        self.generator_optim.zero_grad()
        self.discriminator_optim.zero_grad()

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
        self.train_mode()

        epoch_recon_losses = []
        epoch_g_losses = []
        epoch_d_losses = []
        epoch_z_pos_distances = []
        epoch_z_neg_distances = []

        for i, batch_data in enumerate(data_loader):
            self.zero_grad()

            x1_cpu, x2_cpu, x3_cpu = batch_data

            x1 = x1_cpu.to(self.device, dtype=th.float)
            x2 = x2_cpu.to(self.device, dtype=th.float)

            # train the generative model
            z1_id, cov1_mu, cov1_logvar = self.encoder(x1)
            z2_id, cov2_mu, cov2_logvar = self.encoder(x2)

            z1_cov = reparameterize(training=self.training, mu=cov1_mu, logvar=cov1_logvar)
            z2_cov = reparameterize(training=self.training, mu=cov2_mu, logvar=cov2_logvar)

            kl_divergence_loss_1 = kl_divergence(cov1_logvar, cov1_mu).mean()

            x1_hat = self.decoder(z1_id, z1_cov)
            x21_hat = self.decoder(z2_id, z1_cov)

            # reconstruction loss
            recon_loss_1 = F.mse_loss(x1_hat, x1)
            recon_loss_2 = F.mse_loss(x21_hat, x1)
            autoenc_loss = recon_loss_1 + recon_loss_2 + kl_divergence_loss_1
            autoenc_loss.backward()
            self.autoenc_optim.step()

            # generator loss
            # self.zero_grad()
            x1 = x1_cpu.to(self.device, dtype=th.float)
            x3 = x3_cpu.to(self.device, dtype=th.float)
            z1_id, cov1_mu, cov1_logvar = self.encoder(x1)
            z3_id, cov3_mu, cov3_logvar = self.encoder(x3)
            z1_cov = reparameterize(training=self.training, mu=cov1_mu, logvar=cov1_logvar)
            z3_cov = reparameterize(training=self.training, mu=cov3_mu, logvar=cov3_logvar)

            b_size = len(x3)
            real_label = th.full((b_size,), self.real_label, device=self.device)
            x31_hat = self.decoder(z3_id, z1_cov)
            score_x31 = self.discriminator(x31_hat.detach(), x3)
            gen_err_1 = F.binary_cross_entropy(score_x31, real_label)
            gen_err_1.backward()

            zero_cov = th.FloatTensor(z3_cov.shape)
            zero_cov.normal_(0., 1.)
            zero_cov = zero_cov.to(self.device, dtype=th.float)
            x3e_hat = self.decoder(z3_id, zero_cov)
            score_x3e = self.discriminator(x3e_hat.detach(), x3)
            gen_err_2 = F.binary_cross_entropy(score_x3e, real_label)
            gen_err_2.backward()
            self.autoenc_optim.step()

            # train the discriminator
            ## real class:
            x1 = x1_cpu.to(self.device, dtype=th.float)
            x2 = x2_cpu.to(self.device, dtype=th.float)

            b_size = len(x2)
            real_label = th.full((b_size,), self.real_label, device=self.device)
            score_12 = self.discriminator(x1, x2)
            d_err_real = F.binary_cross_entropy(score_12, real_label)
            d_err_real.backward()

            fake_label = th.full((b_size,), self.fake_label, device=self.device)

            _, cov1_mu, cov1_logvar = self.encoder(x1)
            z2_id, _, _ = self.encoder(x2)
            z1_cov = reparameterize(training=self.training, mu=cov1_mu, logvar=cov1_logvar)

            x21_hat = self.decoder(z2_id, z1_cov)

            score_x21 = self.discriminator(x21_hat.detach(), x2)
            d_err_fake = F.binary_cross_entropy(score_x21, fake_label)
            d_err_fake.backward()
            self.discriminator_optim.step()

            epoch_z_pos_distances.append(F.pairwise_distance(z1_id, z2_id).mean().item())
            epoch_z_neg_distances.append(F.pairwise_distance(z1_id, z3_id).mean().item())

            epoch_recon_losses.append((recon_loss_1 + recon_loss_2).item())
            epoch_g_losses.append((gen_err_1 + gen_err_2).item())
            epoch_d_losses.append((d_err_real + d_err_fake).item())

        recon_loss = np.array(epoch_recon_losses).mean()
        g_loss = np.array(epoch_g_losses).mean()
        d_loss = np.array(epoch_d_losses).mean()
        z_pos_distance = np.array(epoch_z_pos_distances).mean()
        z_neg_distance = np.array(epoch_z_neg_distances).mean()

        return recon_loss, g_loss, d_loss, z_pos_distance, z_neg_distance

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
            recon_loss, g_loss, d_loss, z_pos_dist, z_neg_dis = self.train_epoch(data_loader, epoch)

            self.writer.add_scalars('recon_loss', {'train': recon_loss}, epoch)
            self.writer.add_scalars('g_loss', {'train': g_loss}, epoch)
            self.writer.add_scalars('d_loss', {'train': d_loss}, epoch)
            self.writer.add_scalars('z_dist', {'z_pos_dist': z_pos_dist, 'z_neg_dis': z_neg_dis}, epoch)
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

            if verbose is True:
                print(f'Epoch {epoch + 0:0005}/{num_epochs}: recon_loss: {recon_loss:.5f}, '
                      f'g_loss: {g_loss:.5f}, d_loss: {d_loss}, z_pos_dis: {z_pos_dist:.5f}, '
                      f'z_neg_dis: {z_neg_dis:0.5f}.')

    def visualize(self, epoch, dataset, size=32):
        data_loader = DataLoader(dataset=dataset, batch_size=size, shuffle=False,
                                 num_workers=self.num_workers)
        self.eval_mode()
        for i, batch_data in enumerate(data_loader):
            x1_cpu, x2_cpu, x3_cpu = batch_data
            x1 = x1_cpu.to(self.device, dtype=th.float)
            x2 = x2_cpu.to(self.device, dtype=th.float)
            x3 = x3_cpu.to(self.device, dtype=th.float)

            z1_id, cov1_mu, cov1_logvar = self.encoder(x1)
            z2_id, cov2_mu, cov2_logvar = self.encoder(x2)
            z3_id, cov3_mu, cov3_logvar = self.encoder(x3)

            z1_cov = reparameterize(training=self.training, mu=cov1_mu, logvar=cov1_logvar)
            z2_cov = reparameterize(training=self.training, mu=cov2_mu, logvar=cov2_logvar)
            z3_cov = reparameterize(training=self.training, mu=cov3_mu, logvar=cov3_logvar)

            x1_hat = self.decoder(z1_id, z1_cov)
            x21_hat = self.decoder(z2_id, z1_cov)
            x12_hat = self.decoder(z1_id, z2_cov)
            x32_hat = self.decoder(z3_id, z2_cov)
            x3_hat = self.decoder(z3_id, z3_cov)

            self.plot_images(x1, x2, x3, x1_hat, x21_hat, x12_hat, x32_hat, x3_hat, epoch)
            break

    def plot_images(self, x1, x2, x3, x1_hat, x21_hat, x12_hat, x32_hat, x3_hat, epoch):
        x1 = x1.detach().cpu().numpy()
        x2 = x2.detach().cpu().numpy()
        x3 = x3.detach().cpu().numpy()
        x1_hat = x1_hat.detach().cpu().numpy()
        x21_hat = x21_hat.detach().cpu().numpy()
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
            img_batch.append(th.from_numpy(x21_hat[i, :, :, :]))
            img_batch.append(th.from_numpy(x12_hat[i, :, :, :]))
            img_batch.append(th.from_numpy(x32_hat[i, :, :, :]))
            img_batch.append(th.from_numpy(x3_hat[i, :, :, :]))
        image_grid = tv.utils.make_grid(img_batch, nrow=8)
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
