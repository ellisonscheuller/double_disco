import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#encoder
class Encoder(nn.Module):
    def __init__(self, features, hidden_nodes, latent_dim, zero_init_logvar=True):
        super().__init__()
        layers = []
        input_dim = features

        #given hidden nodes, make the hidden layers
        for i, node in enumerate(hidden_nodes):
            layers += [
                #fully connected layer
                nn.Linear(input_dim, node),
                #activation function
                nn.ReLU(inplace=True),
            ]

            #output of this layer will be the input for the next
            input_dim = node
        self.backbone = nn.Sequential(*layers)

        #latend mean vector
        self.fc_mean   = nn.Linear(input_dim, latent_dim)

        #laten log variance
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

        #initialize logvar layer weights to 0 for stability at the start
        if zero_init_logvar:
            nn.init.zeros_(self.fc_logvar.weight)
            nn.init.zeros_(self.fc_logvar.bias)

    #forward pass through the encoder
    def forward(self, x):

        #shared hidden layers
        h = self.backbone(x)

        #mean vector
        z_mean   = self.fc_mean(h)

        #log var vector
        z_logvar = self.fc_logvar(h)
        
        # reparameterize (z=mu+sigma*epsilon)
        std = torch.exp(0.5 * z_logvar)

        #get random value from normal dist. for epsilon
        eps = torch.randn_like(std)

        #calculate latent vector
        z = z_mean + std * eps

        #return everything
        return z_mean, z_logvar, z


#decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_nodes):
        super().__init__()
        assert len(hidden_nodes) >= 1, "decoder needs at least one layer (the output layer)."

        layers = []
        in_dim = latent_dim
        for i, node in enumerate(hidden_nodes):
            linear = nn.Linear(in_dim, node)
            
            #last layer uses small uniform init to prevent blowup (match diptarko code)
            if i == len(hidden_nodes) - 1:
                nn.init.uniform_(linear.weight, -0.05, 0.05)
                nn.init.uniform_(linear.bias, -0.05, 0.05)
            layers.append(linear)

            #apply batch norm + ReLU for all but last layer
            if i != len(hidden_nodes) - 1:
                layers.append(nn.BatchNorm1d(node))
                layers.append(nn.ReLU(inplace=True))

            in_dim = node

        self.net = nn.Sequential(*layers)

    #decodes z back to reconstructed features
    def forward(self, z):
        return self.net(z)


#vae class
class VAE(nn.Module):
    def __init__(self, config):
        """
        config expects:
          - features: int
          - latent_dim: int
          - encoder_config["nodes"]: List[int]
          - decoder_config["nodes"]: List[int]   (last entry is output dim)
          - alpha: float
          - beta: float
        """
        super().__init__()
        self.alpha = float(config["alpha"])
        self.beta  = float(config["beta"])
        self.reco_scale = self.alpha * (1.0 - self.beta)
        self.kl_scale   = self.beta

        features    = config["features"]
        latent_dim  = config["latent_dim"]
        enc_nodes   = config["encoder_config"]["nodes"]
        dec_nodes   = config["decoder_config"]["nodes"]

        self.encoder = Encoder(features, enc_nodes, latent_dim, zero_init_logvar=True)
        self.decoder = Decoder(latent_dim, dec_nodes)

    @staticmethod
    def kl_divergence(z_mean, z_logvar):
        return 0.5 * torch.sum(torch.exp(z_logvar) + z_mean**2 - 1.0 - z_logvar, dim=1)

    def forward(self, x):
        z_mean, z_logvar, z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z_mean, z_logvar, z

    def compute_losses(self, x, target, reduction="mean", recon_loss_fn=None):
        """
        recon_loss_fn: function taking (pred, target) -> per-sample loss (sum over features)
                       If None, uses MSE summed over features.
        reduction: "mean" or "sum"
        """
        recon, z_mean, z_logvar, _ = self.forward(x)

        if recon_loss_fn is None:
            # MSE summed over features per sample
            recon_per_elem = (recon - target) ** 2
            recon_per_sample = torch.sum(recon_per_elem, dim=1)
        else:
            recon_per_sample = recon_loss_fn(recon, target)

        kl_per_sample = VAE.kl_divergence(z_mean, z_logvar)

        #scale like in vae legacy
        recon_loss = self.reco_scale * recon_per_sample
        kl_loss    = self.kl_scale   * kl_per_sample
        total      = recon_loss + kl_loss

        if reduction == "mean":
            return total.mean(), recon_loss.mean(), kl_loss.mean()
        elif reduction == "sum":
            return total.sum(), recon_loss.sum(), kl_loss.sum()
        else:
            # no reduction: return per-sample terms
            return total, recon_loss, kl_loss
