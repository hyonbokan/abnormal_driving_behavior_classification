import numpy as np
import os
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import cv2                                 
from tqdm import tqdm
import pandas as pd

class_names = ['turning the head', 'blinking', 'falling asleep', 'glancing', 'holding something', 'extending the hand', 
               'being reached by foot', 'tapping the thigh','holding a cellphone','opening the door','standing up', 'spitting',
               'padding the shoulder','clapping','rubbing the eyes','bendding over','turning around','slapping','driving',
               'leaning sideways','looking at something', 'drinking','hands off the wheel', 'immobile', 'touching the neck',
               'looking at the center', 'being reached by hand', 'steering the wheel', 'shaking head', 'massaging an arm', 
               'yawning', 'reaching the middle compartment', 'opening a window', 'shaking something','cellphone at the ear']

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)
print(nb_classes)

IMAGE_SIZE = (150, 150)


def load_data():
    datasets = ['/media/Data-B/data/main_data/train_ppr_cat', '/media/Data-B/data/main_data/val_ppr_cat']
    output = []

    for dataset in datasets:
        images = []
        labels = []

        print('Loading {}'.format(dataset))
        for folder in os.listdir(dataset):
            label = class_names_label[folder]
            # print(label)
            # iter through each imag in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):

                # get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)
                # print(img_path)

                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))

    return output


load_data()


(train_images, train_labels), (test_images, test_labels) = load_data()

n_train = train_labels.shape[0]
n_test = test_labels.shape[0]

print('Number of training samples: {}'.format(n_train))
print('Number of test samples: {}'.format(n_test))
print('Each img is of size: {}'.format(IMAGE_SIZE))