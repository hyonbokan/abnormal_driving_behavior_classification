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

    validate_dataset = datasets.ImageFolder(root="/home/dnlab/Data-B/data/main_data/val_cat_new",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    # net = ResNet(block=BasicBlock, num_classes=5, blocks_num=[2,2,2,2])
    net = resnet34(num_classes=7)
    # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 5) # it could be the number of classes
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 50
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        running_corrects = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # calculate statistics
            running_loss += loss.item()
            _, preds = torch.max(logits, 1)
            running_corrects += torch.sum(preds == labels.to(device))

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                    epochs,
                                                                    loss)

        # calculate train accuracy
        train_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(running_loss / train_steps)
        train_accuracies.append(train_acc.item())

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_loss = 0.0
        val_corrects = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == val_labels.to(device))

        val_acc = acc / val_num
        val_loss /= len(validate_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print('[epoch %d] train_loss: %.3f train_acc: %.3f val_loss: %.3f val_accuracy: %.3f' %
            (epoch + 1, train_losses[-1], train_accuracies[-1], val_loss, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save(net.state_dict(), save_path)

    print('Finished Training')
    print(f"Train loss: {train_losses}")
    print(f"Train_acc: {train_accuracies}")
    print(f"Val loss: {val_losses}")
    print(f"Val_acc: {val_accuracies} ")


if __name__ == '__main__':
    main()
