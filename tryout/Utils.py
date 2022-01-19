import torch.nn as nn
import numpy as np

def uniform_params_initialization(layer):
    f = 1./np.sqrt(layer.weight.data.size()[0])
    nn.init.uniform_(layer.weight.data, -f, f)
    nn.init.uniform_(layer.bias.data, -f, f)