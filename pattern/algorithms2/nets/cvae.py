import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms2.nets.base import BaseVAE

class CVAE(BaseVAE):
    """
    Conditional variational autoencoder.

    For emulator, configs are
        P_GU, P_CGU, P_ABS,
        x = P_GU (s) P_ABS, y = P_CGU,
        ysx = y (s) x = P_CGU (s) P_GU (s) P_ABS
    
    For policy, configs are
        P_GU, P_ABS,
        x = P_GU, y = P_ABS*,
        ysx = y (s) x = P_ABS* (s) x
    """
    def __init__(self,
                 name,
                 x_in_channels,
                 y_in_channels,
                 embedder_latent_dim,
                 encoder_latent_dim,
                 hidden_dims=None):
        super(CVAE, self).__init__()

        assert name in ['emulator', 'policy']
        self.name = name
        self.embedder_latent_dim = embedder_latent_dim
        self.encoder_latent_dim  = encoder_latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
            self.hidden_dims = hidden_dims

        # build embedder for x
        in_channels = x_in_channels
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ))
            in_channels = h_dim
        
        self.embedder = nn.Sequential(*modules)
        self.embedding_to_latent = nn.Linear(hidden_dims[-1]*4, embedder_latent_dim)

        # build encoder for ysx
        modules = []
        in_channels = x_in_channels + y_in_channels
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ))
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, encoder_latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, encoder_latent_dim)

        # build decoder for z
        modules = []
        self.decoder_input = nn.Linear(embedder_latent_dim + encoder_latent_dim, hidden_dims[-1]*4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1],
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.LeakyReLU()
            ))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def embed(self, *tensors):
        """
        For emulator, it needs to pass in (as ordered)
        :param P_GU : [N x 1 x H x W]
        :param P_ABS: [N x 1 x H x W]
        For policy, it only needs to pass in
        :param P_GU : [N x 1 x H x W]
        """
        x = torch.cat(tensors, dim=1)
        result = self.embedder(x)
        result = torch.flatten(result, start_dim=1)
        embedded_latent = self.embedding_to_latent(result)

        return embedded_latent

    def encode(self, *args):
        """
        For emulator, it needs to pass in (as ordered)
        :param P_CGU: [N x 1 x H x W]
        :param P_GU : [N x 1 x H x W]
        :param P_ABS: [N x 1 x H x W]
        For policy, it needs to pass in (as ordered)
        :param P_ABS: [N x 1 x H x W]
        :param P_GU : [N x 1 x H x W]
        """
        ysx = torch.cat(args, dim=1)
        
        result = self.encoder(ysx)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, embedded_latent, other_latent):
        """
        :param embedded_latent: [N x embedder_latent_dim]
        :param    other_latent: [N x  encoder_latent_dim]
        """
        z = torch.cat([embedded_latent, other_latent], dim=1)
        result = self.decoder_input(z)
        result = result.view(-1, 256, 2, 2)
        result = self.decoder(result)
        y_hat_raw = self.final_layer(result)

        return y_hat_raw
    
    def forward(self, *args):
        """
        Feed forward function for training procedure (mainly for computing loss function)

        For emulator, it needs to pass in
        :param P_GU : [N x 1 x H x W]
        :param P_ABS: [N x 1 x H x W]
        :param P_CGU: [N x 1 x H x W]
        For policy, it needs to pass in
        :param P_GU : [N x 1 x H x W]
        :param P_ABS: [N x 1 x H x W]
        """
        if self.name == 'emulator': # emulator
            P_GU, P_ABS = args
        elif self.name == 'policy': # policy
            P_GU, P_ABS, P_CGU = args

        # embedding
        embedded_latent = self.embed(P_GU, P_ABS)

        # encoding
        mu, log_var = self.encode(P_GU, P_ABS, P_CGU)
        encoded_latent = self.reparameterize(mu, log_var)

        # decoding
        y_hat_raw = self.decode(embedded_latent, encoded_latent)
        
        return args + [y_hat_raw, mu, log_var]

    def loss_function(self, *args, **kwargs):
        """
        """
        if self.name == 'emulator': # emulator
            P_GU, P_ABS, P_CGU, y_hat_raw, mu, log_var = args
            mask = P_GU.gt(0.)
            y_hat = y_hat_raw * mask # filtered y_hat
        elif self.name == 'policy': # policy
            P_GU, P_ABS, y_hat_raw, mu, log_var = args
            y_hat = y_hat_raw

        # recons loss
        recons_loss = F.mse_loss(y_hat, P_CGU)

        # KL divergence loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld_weight = kwargs['kld_weight']

        # total loss
        loss = recons_loss + kld_weight * kld_loss

        return {
            'loss': loss,
            'recons_loss': recons_loss,
            'kld_loss': -kld_loss
        }

    def predict(self, *args):
        """
        Provided conditional input x, along with noise from Normal distribution,
        to predict output y_hat.
        For emulator,
        :param P_GU : [N x 1 x H x W]
        :param P_ABS: [N x 1 x H x W]
        For policy,
        :param P_GU : [N x 1 x H x W]
        """
        P_GU = args[0]
        batch_size = P_GU.size()[0]

        # embedded latent
        embedded_latent = self.embed(args)

        # Gaussian noise latent
        noisy_latent = torch.randn(batch_size, self.encoder_latent_dim)

        y_hat_raw = self.decode(embedded_latent, noisy_latent)
        if self.name == 'emulator':
            mask = P_GU.gt(0.)
            y_hat = y_hat_raw * mask
        elif self.name == 'policy':
            y_hat = y_hat_raw
        
        return y_hat

