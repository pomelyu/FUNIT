from abc import ABCMeta, abstractclassmethod
from pathlib import Path
import torch
from models.helper import create_lr_scheduler

class BaseModel(metaclass=ABCMeta):
    def __init__(self, isTrain, device):
        self.isTrain = isTrain
        self.device = device
        self.models = []
        self.losses = []
        self.visuals = []
        self.optimizers = []
        self.lr_schedulers = []
        self.image_paths = []

    @abstractclassmethod
    def set_input(cls, data):
        pass

    @abstractclassmethod
    def forward(cls):
        pass

    @abstractclassmethod
    def optimize_parameters(cls):
        pass

    @abstractclassmethod
    def get_test_outputs(cls):
        pass

    @abstractclassmethod
    @torch.no_grad()
    def evaluate(cls):
        pass

    @torch.no_grad()
    def test(self):
        self.forward()

    def eval(self):
        for name in self.models:
            net = getattr(self, 'net' + name)
            net.eval()

    def initialize_scheduler(self):
        self.lr_schedulers = [create_lr_scheduler(getattr(self, optim_name)) for optim_name in self.optimizers]

    def check_attributes(self):
        for loss_name in self.losses:
            assert hasattr(self, loss_name)
        for model_name in self.models:
            assert hasattr(self, model_name)
        for visual_name in self.visuals:
            assert hasattr(self, visual_name)
        for optimizer_name in self.optimizers:
            assert hasattr(self, optimizer_name)

    def get_losses(self):
        return {loss_name: getattr(self, loss_name).item() for loss_name in self.losses}

    def get_visuals(self):
        return {vis_name: getattr(self, vis_name) for vis_name in self.visuals}

    def get_image_paths(self):
        return self.image_paths

    def update_epoch(self, epoch):
        for scheduler in self.lr_schedulers:
            scheduler.step()
        lr = getattr(self, self.optimizers[0]).param_groups[0]['lr']
        print('learning rate = {:.7f}'.format(lr))

    def save_networks(self, path, epoch="latest", with_optimizer=False):
        for model_name in self.models:
            save_name = Path(path) / "{}_{}.pth".format(epoch, model_name)
            torch.save(getattr(self, model_name).state_dict(), save_name)

        if not with_optimizer:
            return

        for optimizer_name in self.optimizers:
            save_name = Path(path) / "{}_{}.pth".format(epoch, optimizer_name)
            torch.save(getattr(self, optimizer_name).state_dict(), save_name)

    def load_networks(self, path, epoch="latest"):
        for model_name in self.models:
            load_name = Path(path) / "{}_{}.pth".format(epoch, model_name)
            getattr(self, model_name).load_state_dict(torch.load(load_name))
            print("load {}_{}.pth".format(epoch, model_name))

        if not self.isTrain:
            return

        for optimizer_name in self.optimizers:
            load_name = Path(path) / "{}_{}.pth".format(epoch, optimizer_name)
            getattr(self, optimizer_name).load_state_dict(torch.load(load_name))
            print("load {}_{}.pth".format(epoch, optimizer_name))

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
