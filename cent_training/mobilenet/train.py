import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import urllib
from tqdm import tqdm

from model_v2 import MobileNetV2
from model_v3 import MobileNetV3


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

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 8
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


    # download pre-trained weights
    url = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
    model_weight_path = "./mobilenet_v2.pth"
    urllib.request.urlretrieve(url, model_weight_path)

    # load weights into model
    net = MobileNetV2(num_classes=5)
    pre_weights = torch.load(model_weight_path, map_location='cpu')
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # Modify the classifier to have 5 outputs instead of 1000
    num_ftrs = net.classifier[1].in_features
    net.classifier[1] = torch.nn.Linear(num_ftrs, 5)

    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 20
    best_acc = 0.0
    save_path = './MobileNetV2.pth'
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
        train_accuracies.append(train_acc)

        # validate
        # Validate
        net.eval()
        val_loss = 0.0
        val_corrects = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == val_labels.to(device))

        # Calculate validation accuracy
        val_acc = val_corrects.double() / len(validate_loader.dataset)
        val_loss /= len(validate_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print('[Epoch %d/%d] Train Loss: %.3f, Train Acc: %.3f, Val Loss: %.3f, Val Acc: %.3f' %
            (epoch + 1, epochs, train_losses[-1], train_accuracies[-1], val_losses[-1], val_accuracies[-1]))

        # Save best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
    print(f"Train loss: {train_losses}")
    print(f"Train_acc: {train_accuracies}")
    print(f"Val loss: {val_losses}")
    print(f"Val_acc: {val_accuracies} ")


if __name__ == '__main__':
    main()
