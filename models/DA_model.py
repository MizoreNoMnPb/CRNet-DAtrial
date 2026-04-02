import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
from . import losses as L
import torch.nn.functional as F
import torchvision.ops as ops
from util.util import mu_tonemap


# For cityscape Task
class DAModel(BaseModel):
    