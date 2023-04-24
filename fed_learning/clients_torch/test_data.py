import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets





BATCH_SIZE = 8
NUM_CLIENTS = 2

def load_datasets():
    # Download and transform CIFAR-10 (train and test). Images need to be sorted by their classes
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset =  datasets.ImageFolder(root="/media/Data-B/abnormal_drivers_sample/img_category",
                                     transform=transform)
    valset =  datasets.ImageFolder(root="/media/Data-B/abnormal_drivers_sample/img_category",
                                         transform=transform)
    return trainset, valset

def split_datasets(trainset, valset, num_clients):
    # Calculate the number of samples per client
    num_train_samples = len(trainset)
    num_val_samples = len(valset)
    train_samples_per_client = num_train_samples // num_clients
    val_samples_per_client = num_val_samples // num_clients
    
    # Calculate the starting and ending indices for each client
    train_indices = [i * train_samples_per_client for i in range(num_clients)]
    train_indices.append(num_train_samples)
    val_indices = [i * val_samples_per_client for i in range(num_clients)]
    val_indices.append(num_val_samples)
    
    # Split the trainset and valset for each client
    trainloaders = []
    valloaders = []
    for i in range(num_clients):
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=BATCH_SIZE,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                range(train_indices[i], train_indices[i+1])
            ),
            num_workers=2
        )
        trainloaders.append(trainloader)
        
        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=BATCH_SIZE,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                range(val_indices[i], val_indices[i+1])
            ),
            num_workers=2
        )
        valloaders.append(valloader)
        
    return trainloaders, valloaders

trainset, valset = load_datasets()
trainloaders, valloaders = split_datasets(trainset, valset, NUM_CLIENTS)