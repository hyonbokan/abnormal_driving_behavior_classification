import torch
from torchvision import datasets, transforms
import os


train_path = '/home/dnlab/Data-B/data/main_data/train_cat_new'
val_path = '/home/dnlab/Data-B/data/main_data/val_cat_new'

BATCH_SIZE = 32
NUM_CLIENTS = 2
nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])

def load_datasets(train_dir, val_dir):
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    train_dataset = datasets.ImageFolder(root=train_dir,
                                         transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=val_dir,
                                            transform=data_transform["val"])
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=nw)
    return train_loader, validate_loader


def split_dataset(train_loader, validate_loader, num_clients):
    train_datasets = []
    validate_datasets = []
    train_size = len(train_loader.dataset)
    val_size = len(validate_loader.dataset)
    train_indices = list(range(train_size))
    val_indices = list(range(val_size))
    num_train_samples_per_client = train_size // num_clients
    num_val_samples_per_client = val_size // num_clients
    for i in range(num_clients):
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            train_indices[i*num_train_samples_per_client:(i+1)*num_train_samples_per_client])
        validate_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            val_indices[i*num_val_samples_per_client:(i+1)*num_val_samples_per_client])
        train_datasets.append(torch.utils.data.DataLoader(train_loader.dataset,
            batch_size=train_loader.batch_size, sampler=train_sampler))
        validate_datasets.append(torch.utils.data.DataLoader(validate_loader.dataset,
            batch_size=validate_loader.batch_size, sampler=validate_sampler))
    return train_datasets, validate_datasets



train_loader, val_loader = load_datasets(train_path, val_path)
train_sets, val_sets = split_dataset(train_loader, val_loader, NUM_CLIENTS)

print(len(train_sets[0]))

