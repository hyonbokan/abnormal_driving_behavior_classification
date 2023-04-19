import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets





BATCH_SIZE = 36
NUM_CLIENTS = 2

def load_datasets():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset =  datasets.ImageFolder(root="/media/Data-B/data/main_data/train_ppr_cat",
                                     transform=transform)
    valset =  datasets.ImageFolder(root="/media/Data-B/data/main_data/val_ppr_cat",
                                         transform=transform)
    return trainset, valset

# def split_datasets(trainset, testset):
#     partition_size = len(trainset) // NUM_CLIENTS
#     lengths = [partition_size] * NUM_CLIENTS
#     datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
#     # print(datasets)

#     trainloaders = []
#     valloaders = []
#     for ds in datasets:
#         len_val = len(ds) // 10  # 10 % validation set
#         len_train = len(ds) - len_val
#         lengths = [len_train, len_val]
#         ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
#         trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
#         valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
#     testloader = DataLoader(testset, batch_size=BATCH_SIZE)
#     return trainloaders, valloaders, testloader


# print(trainset, testset)
# trainloaders, valloaders, testloader = split_datasets(trainset, testset)

# ----------------------------
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

# print(len(trainloaders[0]))
# print(len(trainloaders[1]))
# print(len(trainloaders[2]))
# print(len(trainloaders[3]))

# num_train_samples = len(trainset)
# num_val_samples = len(valset)
# train_samples_per_client = num_train_samples // 4
# val_samples_per_client = num_val_samples // 4

# train_indices = [i * train_samples_per_client for i in range(4)]
# train_indices.append(num_train_samples)
# val_indices = [i * val_samples_per_client for i in range(4)]
# val_indices.append(num_val_samples)

# print(train_indices)
# print(val_indices)


# trainloaders = []
# valloaders = []
# for i in range(4):
#     trainloader = torch.utils.data.DataLoader(
#         trainset,
#         batch_size=BATCH_SIZE, ## batch_size makes it 12k
#         sampler=torch.utils.data.sampler.SubsetRandomSampler(
#             range(train_indices[i], train_indices[i+1])
#         ),
#         num_workers=2
#     )
#     trainloaders.append(trainloader)
#     print(len(trainloaders[i]))
#     valloader = torch.utils.data.DataLoader(
#         valset,
#         batch_size=BATCH_SIZE,
#         sampler=torch.utils.data.sampler.SubsetRandomSampler(
#             range(val_indices[i], val_indices[i+1])
#         ),
#         num_workers=2
#     )
#     valloaders.append(valloader)

# print(len(trainloaders[0]))
# print(len(trainloaders[1]))
# print(len(trainloaders[2]))
# print(len(trainloaders[3]))