import torch
from torch import nn
from models.components import ConvBlock, DeConvBlock, ResBlock, FlattenLayer

class FUNIT(nn.Module):
    def __init__(self):
        super(FUNIT, self).__init__()
        self.ContentEncoder = nn.Sequential(
            # (3, 128, 128)
            ConvBlock(3, 64, 3, 1, 1, norm="instance", activation="relu"),
            # (64, 128, 128)
            ConvBlock(64, 128, 3, 2, 1, norm="instance", activation="relu"),
            # (128, 64, 64)
            ConvBlock(128, 256, 3, 2, 1, norm="instance", activation="relu"),
            # (256, 32, 32)
            ConvBlock(256, 512, 3, 2, 1, norm="instance", activation="relu"),
            # (512, 16, 16)
            ResBlock(512, norm="instance"),
            ResBlock(512, norm="instance"),
            # (512, 16, 16)
        )

        self.ClassEncoder = nn.Sequential(
            # (3, 128, 128)
            ConvBlock(3, 64, 3, 1, 1, norm="none", activation="relu"),
            # (64, 128, 128)
            ConvBlock(64, 128, 3, 2, 1, norm="none", activation="relu"),
            # (128, 64, 64)
            ConvBlock(128, 256, 3, 2, 1, norm="none", activation="relu"),
            # (256, 32, 32)
            ConvBlock(256, 512, 3, 2, 1, norm="none", activation="relu"),
            # (512, 16, 16)
            ConvBlock(512, 1024, 3, 2, 1, norm="none", activation="relu"),
            # (1024, 8, 8 )
            nn.AvgPool2d(2, 2, 0),
            # (1024, 4, 4)
            FlattenLayer(),
            # 16384
        )

        self.AdaInDecoder = AdaInDecoder(512, 16384)

    def forward(self, x, ys):
        print(len(ys))
        x_content = self.ContentEncoder(x)
        class_codes = []
        for y in ys:
            class_codes.append(self.ClassEncoder(y))
        class_codes = torch.stack(class_codes, dim=0)
        class_code = torch.mean(class_codes, dim=0)

        res = self.AdaInDecoder(x_content, class_code)
        return res


# Ref from https://github.com/NVlabs/MUNIT/blob/master/networks.py
class AdaInDecoder(nn.Module):
    def __init__(self, input_nc, latent_size):
        super(AdaInDecoder, self).__init__()
        self.adain_generator = nn.Sequential(
            # (512, 16, 16)
            ResBlock(input_nc, norm="adaIn"),
            ResBlock(input_nc, norm="adaIn"),
            # (512, 16, 16)
            DeConvBlock(input_nc, 256, norm="instance", activation="relu"),
            # (256, 32, 32)
            DeConvBlock(256, 128, norm="instance", activation="relu"),
            # (128, 64, 64)
            DeConvBlock(128, 64, norm="instance", activation="relu"),
            # (64, 128, 128)
            ConvBlock(64, 3, 3, 1, 1, norm="instance", activation="tanh")
        )

        nc_adain_params = self.get_num_adain_params(self.adain_generator)

        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, nc_adain_params),
        )

    def forward(self, x, class_code):
        adain_params = self.latent_encoder(class_code)
        self.assign_adain_params(adain_params, self.adain_generator)
        x = self.adain_generator(x)
        return x

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params
