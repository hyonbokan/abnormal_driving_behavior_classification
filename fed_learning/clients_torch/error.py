from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import flwr as fl
from flwr.common import Metrics
from resnet import ResNet, BasicBlock
from split_data import trainloaders, valloaders


DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    state_dict = net.state_dict()
    for key, param in zip(state_dict.keys(), parameters):
        if key in state_dict:
            state_dict[key] = torch.tensor(param)
    net.load_state_dict(state_dict, strict=True)

            


def initialize_weights(module):
    if isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


net = ResNet(block=BasicBlock, num_classes=5, blocks_num=[2,2,2,2])
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 5)
net.to(DEVICE)

parameters = get_parameters(net)

params_dict = zip(net.state_dict().keys(), parameters)

# print(params_dict)

state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

# print(state_dict)


state_dict_keys = set(state_dict.keys())
net_state_dict_keys = set(net.state_dict().keys())

print("Keys in state_dict but not in net.state_dict():", state_dict_keys - net_state_dict_keys)
print("Keys in net.state_dict() but not in state_dict:", net_state_dict_keys - state_dict_keys)

new_params = set_parameters(net, parameters=parameters)
print(new_params)
