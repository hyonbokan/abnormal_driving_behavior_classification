from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import flwr as fl
from flwr.common import Metrics
from resnet import ResNet, BasicBlock, resnet34
# from split_data import train_sets, val_sets
from dataloader import train_loader, val_loader
from tqdm import tqdm

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)




def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    net.train()
    for epoch in range(epochs):
        # train
        for images, labels in tqdm(trainloader):
                    optimizer.zero_grad()
                    criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
                    optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


net = resnet34(num_classes=5).to(DEVICE)
# net = ResNet(block=BasicBlock, num_classes=5, blocks_num=[2,2,2,2]).to(DEVICE)
# in_channel = net.fc.in_features
# net.fc = nn.Linear(in_channel, 5)
trainloader = train_loader[1]
testloader = val_loader[1]

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        # Can I have it show the train acc and loss in the {}?
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        # Can make it show the loss here?
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)
