from torch import optim
from torch.optim import lr_scheduler
from gin import config

config.external_configurable(optim.Adam, 'optim.Adam')

config.external_configurable(lr_scheduler.StepLR, 'lr_scheduler.StepLR')
