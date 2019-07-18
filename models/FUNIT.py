import math
import torch
from torch import nn
import torch.nn.functional as F
import gin
from models.BaseModel import BaseModel
from models.components import ConvBlock, DeConvBlock, ResBlock, FlattenLayer
from models.helper import create_network, create_optimizer

# pylint: disable=arguments-differ

@gin.configurable(blacklist=["isTrain", "device"])
class FUNITModel(BaseModel):
    def __init__(self, isTrain, device, lambda_idt=0.1, lambda_feat=1, lambda_gp=10, lr_G=0.0001, lr_D=0.0004, clip_grad=0):
        super(FUNITModel, self).__init__(isTrain, device)

        self.models = ["netG"]
        self.losses = ["loss_G", "loss_D", "loss_idt", "loss_feat", "loss_gp"]
        self.visuals = ["real_A", "real_B", "real_A_pg", "fake_A", "fake_AB"]
        self.loss_G = self.loss_D = self.loss_idt = self.loss_feat = self.loss_gp = None
        self.real_A = self.real_B = self.fake_A = self.fake_AB = self.real_A_pg = self.real_B_pg = None
        self.label_A = self.label_B = None

        # 0 -> 16, 1 -> 32, 2 -> 64, 3 -> 128
        self.pg_index = 0
        self.lambda_idt = lambda_idt
        self.lambda_feat = lambda_feat
        self.lambda_gp = lambda_gp
        self.clip_grad = clip_grad
        self.netG = create_network(FUNIT, device)

        if self.isTrain:
            self.models += ["netD"]
            self.netD = create_network(FUNIT_Dis, device)

            self.criterionIdt = nn.L1Loss(reduction="sum")
            self.optimizerG = create_optimizer(self.netG.parameters(), lr=lr_G)
            self.optimizerD = create_optimizer(self.netD.parameters(), lr=lr_D)
            self.optimizers = ["optimizerG", "optimizerD"]
            self.initialize_scheduler()

        self.check_attributes()

    def set_input(self, input_data): # pylint: disable=arguments-differ
        self.real_A = input_data["source"].to(self.device)
        self.real_B = input_data["target"].to(self.device)

        if self.pg_index <= 0:
            self.real_A_pg = F.interpolate(self.real_A, scale_factor=0.125).detach()
            self.real_B_pg = F.interpolate(self.real_B, scale_factor=0.125).detach()
        elif self.pg_index <= 1:
            self.real_A_pg = F.interpolate(self.real_A, scale_factor=0.25).detach()
            self.real_B_pg = F.interpolate(self.real_B, scale_factor=0.25).detach()
        elif self.pg_index <= 2:
            self.real_A_pg = F.interpolate(self.real_A, scale_factor=0.5).detach()
            self.real_B_pg = F.interpolate(self.real_B, scale_factor=0.5).detach()
        else:
            self.real_A_pg = self.real_A
            self.real_B_pg = self.real_B

        B = self.real_A.shape[0]
        self.label_A = input_data["source_label"].long().view(B, 1, 1, 1).to(self.device)
        self.label_B = input_data["target_label"].long().view(B, 1, 1, 1).to(self.device)

    def forward(self):
        self.fake_A = self.netG(self.real_A, [self.real_A])
        self.fake_AB = self.netG(self.real_A, [self.real_B])

    def backward_G(self):
        _, feat_real_B = self.netD.forward(self.real_B_pg, with_feature=True)
        score_fake_AB, feat_fake_AB = self.netD.forward(self.fake_AB, with_feature=True)

        # Feature match loss
        self.loss_feat = self.criterionIdt(feat_real_B, feat_fake_AB) / feat_real_B.numel()
        self.loss_feat *= self.lambda_feat

        # GAN HingeLoss
        H, W = score_fake_AB.shape[-2:]
        index_B = self.label_B.expand(-1, 1, H, W) # B x D x H x W
        score_fake_AB = torch.gather(score_fake_AB, dim=1, index=index_B)
        self.loss_G = -score_fake_AB.mean()

        self.loss_idt = self.criterionIdt(self.real_A_pg, self.fake_A)
        self.loss_idt = (self.loss_idt / self.real_A_pg.numel()) * self.lambda_idt

        loss = self.loss_G + self.loss_idt + self.loss_feat
        loss.backward()

    def backward_D(self):
        fake_AB = self.netG(self.real_A, [self.real_B])

        if self.lambda_gp != 0:
            self.real_A_pg.requires_grad = True

        score_real_A = self.netD(self.real_A_pg)
        score_fake_AB = self.netD(fake_AB)

        B, _, H, W = score_real_A.shape
        index_A = self.label_A.expand(-1, 1, H, W) # B X 1 x H x W
        index_B = self.label_B.expand(-1, 1, H, W) # B x 1 x H x W
        score_real_A = torch.gather(score_real_A, dim=1, index=index_A)
        score_fake_AB = torch.gather(score_fake_AB, dim=1, index=index_B)

        # Gradient Penality on real data
        if self.lambda_gp == 0:
            self.loss_gp = torch.zeros(1).to(self.device)
        else:
            grad_outputs = torch.ones_like(score_real_A, device=self.device)
            grad_outputs.requires_grad = False
            grad = torch.autograd.grad(outputs=score_real_A, inputs=self.real_A_pg, \
                                        grad_outputs=grad_outputs, only_inputs=True, \
                                        create_graph=True, retain_graph=True)[0]
            grad = grad.view(B, -1)
            self.loss_gp = (grad ** 2).sum(1).mean()
            self.loss_gp *= self.lambda_gp

        self.loss_D = F.relu(1 - score_real_A).mean() + F.relu(1 + score_fake_AB).mean()
        self.loss_D = self.loss_D / 2
        loss = self.loss_D + self.loss_gp
        loss.backward()

    def optimize_parameters(self):
        clip_grad = self.clip_grad

        self.forward()
        self.optimizerG.zero_grad()
        self.backward_G()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), clip_grad)
            torch.nn.utils.clip_grad_norm_(self.netD.parameters(), clip_grad)
        self.optimizerG.step()

        self.optimizerD.zero_grad()
        self.backward_D()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), clip_grad)
            torch.nn.utils.clip_grad_norm_(self.netD.parameters(), clip_grad)
        self.optimizerD.step()

    def get_test_outputs(self):
        pass

    def evaluate(self):
        pass

    def set_pg_index(self, pg_index):
        self.pg_index = pg_index
        self.netG.set_pg_index(self.pg_index)
        self.netD.set_pg_index(self.pg_index)


@gin.configurable()
class FUNIT(nn.Module):
    def __init__(self, dim_class=64):
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
            nn.AdaptiveAvgPool2d(1),
            # (1024, 1, 1)
            ConvBlock(1024, dim_class, 1, 1, 0, norm="none", activation="tanh"),
            # 64
            FlattenLayer(),
        )
        self.AdaInDecoder = AdaInDecoder(512, dim_class)

    def forward(self, x, ys):
        x_content = self.ContentEncoder(x)
        class_codes = []
        for y in ys:
            class_codes.append(self.ClassEncoder(y))
        class_codes = torch.stack(class_codes, dim=0)
        class_code = torch.mean(class_codes, dim=0)
        res = self.AdaInDecoder(x_content, class_code)
        return res

    def set_pg_index(self, pg_index):
        self.AdaInDecoder.set_pg_index(pg_index)


# Ref from https://github.com/NVlabs/MUNIT/blob/master/networks.py
class AdaInDecoder(nn.Module):
    def __init__(self, input_nc, latent_size):
        super(AdaInDecoder, self).__init__()
        # 0 -> 16, 1 -> 32, 2 -> 64, 3 -> 128
        self.pg_index = 0
        self.adain_generator = nn.Sequential(
            # (512, 16, 16)
            ResBlock(input_nc, norm="adaIn"),
            ResBlock(input_nc, norm="adaIn"),
            # (512, 16, 16)
        )

        self.deConv32 = DeConvBlock(input_nc, 256, norm="instance", activation="relu")
        # (256, 32, 32)
        self.deConv64 = DeConvBlock(256, 128, norm="instance", activation="relu")
        # (128, 64, 64)
        self.deConv128 = DeConvBlock(128, 64, norm="instance", activation="relu")
        # (64, 128, 128)

        self.toRGB16 = ConvBlock(512, 3, 3, 1, 1, norm="none", activation="hardtanh")
        self.toRGB32 = ConvBlock(256, 3, 3, 1, 1, norm="none", activation="hardtanh")
        self.toRGB64 = ConvBlock(128, 3, 3, 1, 1, norm="none", activation="hardtanh")
        self.toRGB128 = ConvBlock(64, 3, 3, 1, 1, norm="none", activation="hardtanh")
        self.toRGBs = [self.toRGB16, self.toRGB32, self.toRGB64, self.toRGB128]

        nc_adain_params = self.get_num_adain_params(self.adain_generator)

        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, nc_adain_params),
        )

    def forward(self, x, class_code):
        adain_params = self.latent_encoder(class_code)
        self.assign_adain_params(adain_params, self.adain_generator)
        x = self.adain_generator(x)

        # Progressive growing
        pg_index = math.ceil(self.pg_index)
        reminder = self.pg_index - math.floor(self.pg_index)

        shortcut = None
        if self.pg_index > 0:
            shortcut = x
            x = self.deConv32(x)

        if self.pg_index > 1:
            shortcut = x
            x = self.deConv64(x)

        if self.pg_index > 2:
            shortcut = x
            x = self.deConv128(x)

        RGB = self.toRGBs[pg_index](x)
        if reminder > 0:
            shortcutRGB = self.toRGBs[pg_index - 1](shortcut)
            shortcutRGB = F.interpolate(shortcutRGB, scale_factor=2)
            RGB = (1 - reminder) * shortcutRGB + RGB

        return RGB

    def set_pg_index(self, pg_index):
        self.pg_index = pg_index

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

@gin.configurable()
class FUNIT_Dis(nn.Module):
    def __init__(self, num_class=444, ndf=64):
        super(FUNIT_Dis, self).__init__()
        # 0 -> 16, 1 -> 32, 2 -> 64, 3 -> 128
        self.pg_index = 0
        self.fromRGB128 = nn.Conv2d(3, ndf*1, 3, 1, 1)
        self.fromRGB64 = nn.Conv2d(3, ndf*2, 3, 1, 1)
        self.fromRGB32 = nn.Conv2d(3, ndf*4, 3, 1, 1)
        self.fromRGB16 = nn.Conv2d(3, ndf*8, 3, 1, 1)
        self.fromRGBs = [self.fromRGB16, self.fromRGB32, self.fromRGB64, self.fromRGB128]

        self.conv128 = nn.Sequential(
            FUNITResBlock(ndf*1, ndf*2),
            FUNITResBlock(ndf*2, ndf*2),
            nn.AvgPool2d(2, 2),
        )
        # (128, 64, 64)
        self.conv64 = nn.Sequential(
            FUNITResBlock(ndf*2, ndf*4),
            FUNITResBlock(ndf*4, ndf*4),
            nn.AvgPool2d(2, 2),
        )
        # (256, 32, 32)
        self.conv32 = nn.Sequential(
            FUNITResBlock(ndf*4, ndf*8),
            FUNITResBlock(ndf*8, ndf*8),
            nn.AvgPool2d(2, 2),
        )
        # (512, 16, 16)
        self.conv16 = nn.Sequential(
            FUNITResBlock(ndf*8, ndf*16),
            FUNITResBlock(ndf*16, ndf*16),
            nn.AvgPool2d(2, 2),
        )
        self.convs = [self.conv16, self.conv32, self.conv64, self.conv128]
        # (1024, 8, 8)
        self.conv8 = nn.Sequential(
            FUNITResBlock(ndf*16, ndf*16),
            FUNITResBlock(ndf*16, ndf*16),
        )
        self.conv_out = nn.Conv2d(ndf*16, num_class, 1, 1, 0)

    def forward(self, x, with_feature=False):
        # Progressive shrinking
        pg_index = math.ceil(self.pg_index)
        reminder = self.pg_index - math.floor(self.pg_index)

        shortcut = x
        x = self.fromRGBs[pg_index](x)
        if reminder > 0:
            x = self.convs[pg_index](x)
            shortcut = F.interpolate(shortcut, scale_factor=0.5)
            shortcut = self.fromRGBs[pg_index - 1](shortcut)
            x = (1 - reminder) * shortcut + x

        if self.pg_index >= 3:
            x = self.conv128(x)

        if self.pg_index >= 2:
            x = self.conv64(x)

        if self.pg_index >= 1:
            x = self.conv32(x)

        if self.pg_index >= 0:
            x = self.conv16(x)

        feature = self.conv8(x)
        res = self.conv_out(feature)

        if with_feature:
            return res, feature
        return res

    def set_pg_index(self, pg_index):
        self.pg_index = pg_index


class FUNITResBlock(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(FUNITResBlock, self).__init__()
        self.norm1 = nn.InstanceNorm2d(input_nc)
        self.actv1 = nn.LeakyReLU(True)
        self.conv1 = nn.Conv2d(input_nc, output_nc, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(output_nc)
        self.actv2 = nn.LeakyReLU(True)
        self.conv2 = nn.Conv2d(output_nc, output_nc, 3, 1, 1)
        if input_nc != output_nc:
            self.convs = nn.Conv2d(input_nc, output_nc, 1, 1)

    def forward(self, x):
        x = self.norm1(x)
        x = self.actv1(x)
        s = self.convs(x) if hasattr(self, "convs") else x
        x = self.conv1(x)
        x = self.conv2(self.actv2(self.norm2(x)))
        return x + s
