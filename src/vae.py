import torch
import torch.nn as nn


class FlexibleVAE(nn.Module):
    def __init__(self, latent_dim=1024, input_size=32):
        super(FlexibleVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size  # 输入图像尺寸，可为32或224

        # Encoder: 根据输入尺寸计算特征图尺寸
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # downsample 2
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # downsample 2
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # downsample 2
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),# downsample 2
            nn.ReLU(True)
        )

        # 计算encoder输出特征图大小
        def conv_output_size(size, kernel=4, stride=2, padding=1, layers=4):
            for _ in range(layers):
                size = (size - kernel + 2*padding)//stride + 1
            return size
        
        self.feature_map_size = conv_output_size(self.input_size)
        self.fc_mu = nn.Linear(256 * self.feature_map_size**2, latent_dim)
        self.fc_logvar = nn.Linear(256 * self.feature_map_size**2, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * self.feature_map_size**2)

        # Decoder: 对称结构
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 256, self.feature_map_size, self.feature_map_size)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
