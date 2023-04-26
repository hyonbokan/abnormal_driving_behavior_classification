import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import os


train_path = '/home/dnlab/Data-B/websocket/3/dataset/data_balanced/centralized/train'
val_path = '/home/dnlab/Data-B/websocket/3/dataset/data_balanced/centralized/test'

BATCH_SIZE = 32
NUM_CLIENTS = 2
nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])


def load_datasets(train_dir, val_dir, num_clients):
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # Load and split training dataset
    train_dataset = datasets.ImageFolder(root=train_dir,
                                         transform=data_transform["train"])
    print(f'Total train: {len(train_dataset)}') # Should be an even number
    partition_size = len(train_dataset) // num_clients
    lengths = [partition_size] * num_clients
    train_datasets = random_split(train_dataset, lengths, torch.Generator().manual_seed(42))

    # Create train dataloaders for each client
    train_loaders = []
    for train_ds in train_datasets:
        train_loader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=nw)
        train_loaders.append(train_loader)

    # Load and split validation dataset
    validate_dataset = datasets.ImageFolder(root=val_dir,
                                            transform=data_transform["val"])
    print(f'Total val: {len(validate_dataset)}') # Should be an even number
    val_partition_size = len(validate_dataset) // num_clients
    val_lengths = [val_partition_size] * num_clients
    val_datasets = random_split(validate_dataset, val_lengths, torch.Generator().manual_seed(42))

    # Create validation dataloaders for each client
    val_loaders = []
    for val_ds in val_datasets:
        val_loader = torch.utils.data.DataLoader(val_ds,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False,
                                                 num_workers=nw)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders

train_loader, val_loader = load_datasets(train_dir=train_path, val_dir=val_path, num_clients=NUM_CLIENTS)

len(val_loader[0])