from torch import optim
from torch.optim import lr_scheduler
from torch import nn
from gin import config
import gin

# Optimizer
config.external_configurable(optim.Adam, "optim.Adam")
config.external_configurable(optim.Adadelta, "optim.Adadelta")
config.external_configurable(optim.Adagrad, "optim.Adagrad")
config.external_configurable(optim.Adamax, "optim.Adamax")
config.external_configurable(optim.RMSprop, "optim.RMSprop")
config.external_configurable(optim.SGD, "optim.SGD")


# Learning rate decay
config.external_configurable(lr_scheduler.StepLR, "lr_scheduler.StepLR")
config.external_configurable(lr_scheduler.MultiStepLR, "lr_scheduler.MultiStepLR")
config.external_configurable(lr_scheduler.ExponentialLR, "lr_scheduler.ExponentialLR")
config.external_configurable(lr_scheduler.CosineAnnealingLR, "lr_scheduler.CosineAnnealingLR")
config.external_configurable(lr_scheduler.ReduceLROnPlateau, "lr_scheduler.ReduceLROnPlateau")

@gin.configurable("lr_scheduler.LambdaLR", blacklist=["optimizer"])
def create_LambdaLR(optimizer, epoch_total, epoch_decay):
    def lambda_rule(epoch):
        epoch_stable = epoch_total - epoch_decay
        lr_l = 1.0 - max(0, epoch - epoch_stable) / float(epoch_decay + 1)
        return lr_l

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


# Loss
config.external_configurable(nn.L1Loss, "nn.L1Loss")
config.external_configurable(nn.MSELoss, "nn.MSELoss")


# Activateion
config.external_configurable(nn.LeakyReLU, "nn.LeakyReLU")
config.external_configurable(nn.ReLU, "nn.ReLU")
config.external_configurable(nn.ReLU6, "nn.ReLU6")
config.external_configurable(nn.Sigmoid, "nn.Sigmoid")
config.external_configurable(nn.Softplus, "nn.Softplus")
config.external_configurable(nn.Tanh, "nn.Tanh")
