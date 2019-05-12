import itertools
import torch
from torch import nn
import gin
from models.BaseModel import BaseModel
from models.components import ConvBlock, DeConvBlock, FlattenLayer, ReshapeLayer, DeConvLayer
from models.helper import create_network, create_optimizer

# pylint: disable=attribute-defined-outside-init

@gin.configurable(blacklist=["isTrain", "device"])
class VAE(BaseModel):
    def __init__(self, isTrain, device, lambda_kl=10):
        super(VAE, self).__init__(isTrain, device)

        self.models = ["netEncoder", "netDecoder"]
        self.losses = ["loss_Idt", "loss_KL"]
        self.visuals = ["real", "fake"]
        self.loss_Idt = self.loss_KL = None
        self.real = self.fake = None
        self.mu = self.logvar = None

        self.lambda_kl = lambda_kl

        self.netEncoder = create_network(VAEEncoder, device)
        self.netDecoder = create_network(VAEDecoder, device)

        if self.isTrain:
            self.criterionIdt = nn.MSELoss(reduction="sum")
            self.criterionKL = KLLoss()
            params = itertools.chain(self.netEncoder.parameters(), self.netDecoder.parameters())
            self.optimizer = create_optimizer(params)
            self.optimizers = ["optimizer"]

        self.check_attributes()

    def set_input(self, input_data): # pylint: disable=arguments-differ
        self.real = input_data["image"].to(self.device)
        self.image_paths = input_data["path"]

    def forward(self):
        self.mu, self.logvar = self.netEncoder(self.real)
        z = self.reparameterize(self.mu, self.logvar)
        self.fake = self.netDecoder(z)

    def backward(self):
        # loss_KL is calculated on z, (batch_size, latent_size)
        # loss_Idt is calculated on real/fake, (batch_size, depth, width, height)
        # Hence they should be average by batch_size, not by the number of elements
        num_els = self.real.shape[0]
        self.loss_KL = self.lambda_kl * self.criterionKL(self.mu, self.logvar)
        self.loss_Idt = self.criterionIdt(self.fake, self.real) / num_els
        loss = self.loss_Idt + self.loss_KL
        loss.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def get_test_outputs(self):
        pass

    def evaluate(self):
        pass


@gin.configurable()
class VAEEncoder(nn.Module):
    def __init__(self, latent_size, ngf=128, norm="batch", use_bias=False):
        super(VAEEncoder, self).__init__()
        self.model = nn.Sequential(
            # 3, 64, 64 -> 32, 32, 32
            ConvBlock(3, ngf*1, norm=norm, use_bias=use_bias),
            # 32, 32, 32 -> 64, 16, 16
            ConvBlock(ngf*1, ngf*2, norm=norm, use_bias=use_bias),
            # 64, 16, 16 -> 128, 8, 8
            ConvBlock(ngf*2, ngf*4, norm=norm, use_bias=use_bias),
            # 128, 8, 8 -> 256, 4, 4
            ConvBlock(ngf*4, ngf*8, norm=norm, use_bias=use_bias),
            # 256, 4, 4 -> 256, 2, 2
            ConvBlock(ngf*8, ngf*8, norm=norm, use_bias=use_bias),
            # 256, 2, 2 -> 1024
            FlattenLayer(),
        )
        self.mu_output = nn.Linear(1024, latent_size)
        self.logvar_output = nn.Linear(1024, latent_size)

    def forward(self, x):
        x = self.model(x)
        return self.mu_output(x), self.logvar_output(x)


@gin.configurable()
class VAEDecoder(nn.Module):
    def __init__(self, latent_size, ngf=128, norm="batch", use_bias=False):
        super(VAEDecoder, self).__init__()
        self.model = nn.Sequential(
            # latent -> 1024
            nn.Linear(latent_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 1024 -> 256, 2, 2
            ReshapeLayer((256, 2, 2)),
            # 256, 2, 2 -> 256, 4, 4
            DeConvBlock(ngf*8, ngf*8, method="deConv", norm=norm, use_bias=use_bias),
            # 256, 4, 4 -> 128, 8, 8
            DeConvBlock(ngf*8, ngf*4, method="deConv", norm=norm, use_bias=use_bias),
            # 128, 8, 8 -> 64, 16, 16
            DeConvBlock(ngf*4, ngf*2, method="deConv", norm=norm, use_bias=use_bias),
            # 64, 16, 16 -> 32, 32, 32
            DeConvBlock(ngf*2, ngf*1, method="deConv", norm=norm, use_bias=use_bias),
            # 32, 32, 32 -> 3, 64, 64
            DeConvLayer(ngf*1, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class KLLoss(nn.Module):
    def __call__(self, mu, logvar):
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return klds.sum(1).mean(0)
