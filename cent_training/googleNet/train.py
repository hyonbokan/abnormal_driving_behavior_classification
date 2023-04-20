import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from model import GoogLeNet



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    data_transform = {
        "train": transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = datasets.ImageFolder(root="/home/dnlab/Data-B/data/main_data/train_cat_new",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    img_list = train_dataset.class_to_idx
    # print(img_list)
    cla_dict = dict((val, key) for key, val in img_list.items())
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

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    epochs = 30
    best_acc = 0.0
    save_path = './googleNet_batch_8.pth'
    train_steps = len(train_loader)
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in tqdm(range(epochs)):
        # train
        net.train()
        running_loss = 0.0
        # let's tru to skip the line below
        # train_bar = tqdm(train_loader, file=sys.stdout) 
        for step, data in tqdm(enumerate(train_loader)):
            images, labels = data
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            train_loader.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            # val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / train_num
        # Save the metrics
        train_losses.append(running_loss / train_steps)
        val_accuracies.append(val_accurate)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
    print(train_losses)
    print(val_accuracies)


if __name__ == '__main__':
    main()
