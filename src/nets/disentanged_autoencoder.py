import torch as th
from torch import nn
from nets.encoders import Encoder
from nets.decoders import Decoder

class DlrAutoEncoder(nn.Module):
    def __init__(self):
        super(DlrAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x1, x2, x3):
        z1_id, z1_cov = self.encoder(x1)
        z2_id, z2_cov = self.encoder(x2)
        z3_id, z3_cov = self.encoder(x3)

        x1_hat = self.decoder(z1_id, z1_cov)
        x3_hat = self.decoder(z3_id, z3_cov)
        x12_hat = self.decoder(z1_id, z2_cov)
        x32_hat = self.decoder(z3_id, z2_cov)

        z12_id, z12_cov = self.encoder(x12_hat)
        z32_id, z32_cov = self.encoder(x32_hat)

        return x1_hat, x3_hat, x12_hat, x32_hat, z1_id, z1_cov, z2_id, z2_cov, z3_id, z3_cov, \
               z12_id, z12_cov, z32_id, z32_cov