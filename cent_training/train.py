import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import ResNet, BasicBlock, resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    train_dataset = datasets.ImageFolder(root="/home/dnlab/Data-B/data/main_data/train_cat_new",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root="/home/dnlab/Data-B/data/main_data/val_new",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = resnet34(num_classes=7)
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    epochs = 20
    save_path = './resNet34.pth'
    loss_val = []
    acc_val = []

    for epoch in range(epochs):
        net.train()
        # train
        for images, labels in tqdm(train_loader):
                optimizer.zero_grad()
                criterion(net(images.to(device)), labels.to(device)).backward()
                optimizer.step()

        # validate
        net.eval()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for images, labels in tqdm(validate_loader):
                outputs = net(images.to(device))
                labels = labels.to(device)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        accuracy = correct / len(validate_loader.dataset)
        loss /= len(validate_loader)
        loss_val.append(loss)
        acc_val.append(accuracy)

        print('[epoch %d] val_loss: %.3f val_accuracy: %.3f' %
            (epoch + 1, loss ,accuracy))


    print('Finished Training')
    print(f"Val loss: {loss_val}")
    print(f"Val_acc: {acc_val} ")


if __name__ == '__main__':
    main()
