import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
import torch
from util.util import torch_save
import math 
import torch.nn.functional as F
from data.degrade.process import demosaic
from .base_model import BaseModel

class BaseRGBModel(ABC):
    def __init__(self, opt):
        self.base = BaseModel(opt)
        # using base model functions
        
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        
        self.device = self.base.device
        self.save_dir = self.base.save_dir
        
        self.best_metric = float('inf') if self.isTrain else 0
        self.best_epoch = -1
        
        self.losses = []
        self.models = []
        self.visualizers = []
        self.optimizers = []
        self.optimizer_names = []
        self.metric = 0
        self.start_epoch = 0
        self.backwarp_tenGrid = {}
        self.backwarp_tenPartial = {}
        
        self.image_paths = []
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def set_input(self, input):
        pass
    
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def optimize_parameters(self):
        pass
    
    @abstractmethod
    def _save_best_model(self, epoch):
        pass
    
    def setup(self, opt):
        opt = opt if opt is not None else self.opt
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) \
                                for optimizer in self.optimizers]
            for scheduler in self.schedulers:
                scheduler.last_epoch = opt.load_iter
        if opt.load_iter > 0 or opt.load_path != '':
            self.load_networks(opt.load_iter)
            if opt.load_optimizers:
                self.load_optimizers(opt.load_iter)
        
        self.print_networks(opt.verbose)
    
    def eval(self):
        for model in self.models:
            net = getattr(self, 'net' + model)
            net.eval()
    
    def train(self):
        self.isTrain = True
        for model in self.models:
            net = getattr(self, 'net' + model)
            net.train()
    
    def test(self):
        self.isTrain = False
        with torch.no_grad():
            self.forward()
    
    def update_learning_rate(self, epoch):
        for i, scheduler in enumerate(self.schedulers):
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(self.metric)
            elif scheduler.__class__.__name__ == 'CosineLRScheduler':
                scheduler.step(epoch)
            else:
                scheduler.step()
            print('learning rate for {} = {}'.format(self.optimizer_names[i], self.optimizers[i].param_groups[0]['lr']))
                
    def preprocess(self, image):
        if isinstance(image, torch.Tensor):
            image = image * self.rgb_std + self.rgb_mean
            image = torch.clamp(image, 0, 1)
        return image
    
    def get_image_paths(self):
        return self.image_paths
    
    def get_visualizers(self):
        visualizers = OrderedDict()
        if self.isTrain:
            for visualizer in self.visualizers:
                if isinstance(getattr(self, visualizer), list):
                    visualizers[visualizer] = self.preprocess(getattr(self, visualizer)[-1][0:1].detach())
                elif isinstance(getattr(self, visualizer), torch.Tensor):
                    visualizers[visualizer] = self.preprocess(getattr(self, visualizer)[0:1].detach())
                else:
                    raise TypeError(f"Unsupported type for visualizer {visualizer}")
        else:
            for name in self.visualizers:
                visualizers[name] = self.preprocess(getattr(self, name)[0:1].detach())
        return visualizers
            
    def get_losses(self):
        losses = OrderedDict()
        for model in self.models:
            losses[model] = float(getattr(self, 'loss_' + model))
        return losses
    
    def save_networks(self, epoch):
        for model in self.models:
            save_filename = '{}_{}.pth'.format(model, epoch)
            save_path = os.path.join(self.save_dir, 'model', save_filename)
            net = getattr(self, 'net' + model)
            state = {'state_dict': net.module.state_dict() if len(self.gpu_ids) > 0 and torch.cuda.is_available() else net.state_dict()}
            torch_save(state, save_path)
        self.save_optimizers(epoch)
        self._save_best_model(epoch)
    
    def save_optimizers(self, epoch):
        assert len(self.optimizers) == len(self.optimizer_names), "Number of optimizers and optimizer names must match"
        for id, optimizer in enumerate(self.optimizers):
            save_filename = '{}_{}.pth'.format(self.optimizer_names[id], epoch)
            save_path = os.path.join(self.save_dir, 'optimizer', save_filename)
            state = {'state_dict': optimizer.state_dict()}
            torch_save(state, save_path)
    
    def load_networks(self, epoch):
        for model in self.models:
            load_filename = '{}_{}.pth'.format(model, epoch)
            if self.opt.load_path != '':
                load_filename = self.opt.load_path
            else:
                load_path = os.path.join(self.save_dir, 'model', load_filename)
            if not os.path.isfile(load_path):
                print(f"Model file {load_path} does not exist. Skipping loading for {model}.")
                continue
            state = torch.load(load_path, map_location=self.device)
            net = getattr(self, 'net' + model)
            print(f"Loading model {model} from {load_path}")
            
            net_state = net.state_dict()
            is_loaded = {n:False for n in net.state_dict().keys()}
            
            for name, param in state['state_dict'].items():
                if name in state['state_dict']:
                    try:
                        net_state[name].copy_(param)
                        is_loaded[name] = True            
                    except Exception:
                        print(f'While copying the parameter named {name},'
                              f'whose dimensions in the model are {list(net_state[name].shape)} and'
                              f'whose dimensions in the checkpoint are {list(param.shape)}.')
                        raise RuntimeError
                else:
                    print(f'Saved parameter named {name} is not found in the current model.')
            mark = True
            for name in is_loaded:
                if not is_loaded[name]:
                    print(f'Parameter named {name} is not loaded.')
                    mark = False
            if mark:
                print(f'All parameters are initialized using {load_path}.')
            self.start_epoch = epoch
    
    def load_optimizers(self, epoch):
        assert len(self.optimizers) == len(self.optimizer_names), "Number of optimizers and optimizer names must match"
        for id, optimizer in enumerate(self.optimizers):
            load_filename = '{}_{}.pth'.format(self.optimizer_names[id], epoch)
            load_path = os.path.join(self.save_dir, 'optimizer', load_filename)
            if not os.path.isfile(load_path):
                print(f"Optimizer file {load_path} does not exist. Skipping loading for {self.optimizer_names[id]}.")
                continue
            state = torch.load(load_path, map_location=self.device)
            print(f"Loading optimizer {self.optimizer_names[id]} from {load_path}")
            optimizer.load_state_dict(state['state_dict'])
    
    # Functions below are used for multi-input sence, which isn't the task for DA.
    # just used for test.
    def estimate(self, tenFirst, tenSecond, net):
        assert(tenFirst.shape[3] == tenSecond.shape[3])
        assert(tenFirst.shape[2] == tenSecond.shape[2])
        intWidth = tenFirst.shape[3]
        intHeight = tenFirst.shape[2]
        # tenPreprocessedFirst = tenFirst.view(1, 3, intHeight, intWidth)
        # tenPreprocessedSecond = tenSecond.view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tenPreprocessedFirst = F.interpolate(input=tenFirst, 
                                size=(intPreprocessedHeight, intPreprocessedWidth), 
                                mode='bilinear', align_corners=False)
        tenPreprocessedSecond = F.interpolate(input=tenSecond, 
                                size=(intPreprocessedHeight, intPreprocessedWidth), 
                                mode='bilinear', align_corners=False)

        tenFlow = 20.0 * F.interpolate(
                         input=net(tenPreprocessedFirst, tenPreprocessedSecond), 
                         size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tenFlow[:, :, :, :]
    
    def backwarp(self, tenInput, tenFlow):
        index = str(tenFlow.shape) + str(tenInput.device)
        if index not in self.backwarp_tenGrid:
            tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), 
                     tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
            tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), 
                     tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
            self.backwarp_tenGrid[index] = torch.cat([tenHor, tenVer], 1).to(tenInput.device)

        if index not in self.backwarp_tenPartial:
            self.backwarp_tenPartial[index] = tenFlow.new_ones([
                 tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])

        tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), 
                             tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
        tenInput = torch.cat([tenInput, self.backwarp_tenPartial[index]], 1)

        tenOutput = F.grid_sample(input=tenInput, 
                    grid=(self.backwarp_tenGrid[index] + tenFlow).permute(0, 2, 3, 1), 
                    mode='bilinear', padding_mode='zeros', align_corners=False)

        return tenOutput

    def get_backwarp(self, tenFirst, tenSecond, net, flow=None):
        if flow is None:
            flow = self.get_flow(tenFirst, tenSecond, net)
        
        tenoutput = self.backwarp(tenSecond, flow) 	
        tenMask = tenoutput[:, -1:, :, :]
        tenMask[tenMask > 0.999] = 1.0
        tenMask[tenMask < 1.0] = 0.0
        return tenoutput[:, :-1, :, :] * tenMask, tenMask

    def get_flow(self, tenFirst, tenSecond, net):
        with torch.no_grad():
            net.eval()
            flow = self.estimate(tenFirst, tenSecond, net) 
        return flow