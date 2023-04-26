import numpy as np
import os              
import cv2                                            
from tqdm import tqdm
import math
import random


class_names = ['drinking', 'driving', 'falling_asleep', 'turning_around', 'using_cellphone']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (150, 150)
NUM_CLIENTS = 2
train_dataset = '/home/dnlab/Data-B/data/main_data/train_cat_new' 
test_dataset = '/home/dnlab/Data-B/data/main_data/val_cat_new'

def load_data(datasets_url):
    datasets = [datasets_url]
    output = []

    for dataset in datasets:
        images = []
        labels = []

        print('Loading {}'.format(dataset))
        for folder in os.listdir(dataset):
            label = class_names_label[folder]
            # iter through each imag in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):

                # get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)

                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        # output.append((images, labels))

    return images, labels

def split_data(images, labels, NUM_CLIENTS):
    # Shuffle the data
    data = list(zip(images, labels))
    random.shuffle(data)
    images, labels = zip(*data)

    subset_size = math.ceil(len(images) / NUM_CLIENTS)

    client_data = []
    for i in range(NUM_CLIENTS):
        start_index = i * subset_size
        end_index = (i + 1) * subset_size
        images_subset = images[start_index:end_index]
        labels_subset = labels[start_index:end_index]
        client_data.append((images_subset, labels_subset))
    return client_data


images_train, labels_train = load_data(train_dataset)
images_test, labels_test = load_data(test_dataset)

trainset_split = split_data(images_train, labels_train, NUM_CLIENTS)
testset_split = split_data(images_test, labels_test, NUM_CLIENTS)

img, lb = trainset_split[0]
print(len(lb))