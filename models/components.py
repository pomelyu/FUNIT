import functools
import torch
from torch import nn
import torch.nn.functional as F

# pylint: disable=arguments-differ

### Loss ###
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.ones(1) * target_real_label)
        self.register_buffer('fake_label', torch.ones(1) * target_fake_label)
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, x, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(x)

    def __call__(self, x, target_is_real):
        target_tensor = self.get_target_tensor(x, target_is_real)
        return self.loss(x, target_tensor)

### Blocks ###
class DeConvBlock(nn.Sequential):
    def __init__(self, input_nc, output_nc, method="convTrans", \
        kernel_size=4, stride=2, padding=1, norm="batch", activation="lrelu", use_bias=False):
        super(DeConvBlock, self).__init__()

        norm_layer = get_norm_layer(norm)
        actv_layer = get_activation(activation)
        if method == "convTrans":
            self.add_module("deconv", nn.ConvTranspose2d(input_nc, output_nc, kernel_size, \
                                stride, padding=padding, bias=use_bias))
        elif method == "deConv":
            self.add_module("deconv", DeConvLayer(input_nc, output_nc))
        elif method == "pixlSuffle":
            raise NotImplementedError("PixelSuffle not implemente")
        else:
            raise NameError("Unknown method: " + method)

        if norm_layer:
            self.add_module("norm", norm_layer(output_nc))
        if actv_layer:
            self.add_module("actv", actv_layer)

class ConvBlock(nn.Sequential):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1, \
        norm="batch", activation="lrelu", use_bias=False):
        super(ConvBlock, self).__init__()

        norm_layer = get_norm_layer(norm)
        actv_layer = get_activation(activation)
        self.add_module("conv", nn.Conv2d(input_nc, output_nc, kernel_size, stride, \
                            padding=padding, bias=use_bias))
        if norm_layer:
            self.add_module("norm", norm_layer(output_nc))
        if actv_layer:
            self.add_module("actv", actv_layer)


class ResBlock(nn.Module):
    def __init__(self, nc, norm="batch", use_bias=False):
        super(ResBlock, self).__init__()
        self.block1 = ConvBlock(nc, nc, 3, 1, 1, norm=norm, activation="relu", use_bias=use_bias)
        self.block2 = ConvBlock(nc, nc, 3, 1, 1, norm=norm, activation="none", use_bias=use_bias)

    def forward(self, x):
        residual = x
        x = self.block1(x)
        x = self.block2(x)
        x += residual
        return x

### Layers ###
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "adaIn":
        norm_layer = AdaptiveInstanceNorm2d
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError("Unsupported normalization: {}".format(norm_type))
    return norm_layer

def get_pad_layer(pad_type="zero"):
    if pad_type == "zero":
        pad_layer = functools.partial(nn.ZeroPad2d)
    elif pad_type == "reflect":
        pad_layer = functools.partial(nn.ReflectionPad2d)
    elif pad_type == "replicate":
        pad_layer = functools.partial(nn.ReplicationPad2d)
    else:
        raise NotImplementedError("Unsupported padding: {}".format(pad_type))
    return pad_layer

def get_activation(activation="relu"):
    if activation == 'relu':
        activation_layer = nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        activation_layer = nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'prelu':
        activation_layer = nn.PReLU()
    elif activation == 'selu':
        activation_layer = nn.SELU(inplace=True)
    elif activation == 'tanh':
        activation_layer = nn.Tanh()
    elif activation == 'none':
        activation_layer = None
    else:
        raise NotImplementedError('Unsupported activation: {}'.format(activation))
    return activation_layer

class InterpolateLayer(nn.Module):
    def __init__(self, scale_factor):
        super(InterpolateLayer, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)

class DeConvLayer(nn.Module):
    def __init__(self, input_nc, output_nc, use_bias=False):
        super(DeConvLayer, self).__init__()
        self.model = nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        return self.model(x)

class FlattenLayer(nn.Module):
    def forward(self, x):
        num_batch = x.shape[0]
        return x.view(num_batch, -1)

class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        num_batch = x.shape[0]
        return x.view(num_batch, *self.shape)

class L2NormalizeLayer(nn.Module):
    def forward(self, x):
        assert len(x.shape) == 2
        return nn.functional.normalize(x, p=2, dim=1)

class GradientReverseLayer(nn.Module):
    def __init__(self, scale):
        super(GradientReverseLayer, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x.clone()

    def backward(self, grad_out):
        return -self.scale * grad_out.clone()


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
