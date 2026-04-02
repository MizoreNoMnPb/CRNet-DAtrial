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
        
        self.losses = []
        self.models = []
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
    
    def setup(self, opt):
        pass
        
    def post_process(self, image):
        if isinstance(image, torch.Tensor):
            image = image * self.rgb_std + self.rgb_mean
            image = torch.clamp(image, 0, 1)
        return image
    
    def 