from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from flwr.common import Metrics
from resnet import ResNet, BasicBlock, resnet34
from dataloader import train_loader, val_loader
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)
EPOCHS = 1
MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-5

net = resnet34(num_classes=5).to(DEVICE)
trainloader = train_loader[0]
testloader = val_loader[0]


# privacy_engine = PrivacyEngine()
# net, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
#     module=net,
#     optimizer=optimizer,
#     data_loader=trainloader,
#     epochs=EPOCHS,
#     target_epsilon=EPSILON,
#     target_delta=DELTA,
#     max_grad_norm=MAX_GRAD_NORM,
# )

# print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")


def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    # Add DP code here
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
        train(net, trainloader, epochs=EPOCHS)
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