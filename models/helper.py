import torch
from torch.nn import init
import gin

@gin.configurable(blacklist=["optimizer"])
def create_lr_scheduler(optimizer, scheduler=None, **kwargs):
    return scheduler(optimizer, **kwargs)

@gin.configurable(blacklist=["params"])
def create_optimizer(params, optimizer=None, **kwargs):
    return optimizer(params, **kwargs)

@gin.configurable(blacklist=["model", "device"])
def create_network(model, device, init_type="normal", init_gain=0.02, load_path=""):
    model = model()
    if load_path:
        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict)
    else:
        for name, net in model.named_modules():
            if not name or "." in name:
                continue
            init_weights(net, init_type, gain=init_gain)

        print('initialize network with {}'.format(init_type))

    model = model.to(device)
    return model

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
