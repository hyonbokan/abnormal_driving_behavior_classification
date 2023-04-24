import os
import shutil

from sklearn.model_selection import train_test_split
from tqdm import tqdm

input_dir = '/media/Data-B/data/main_data/train_cat_new/turning_around'
train_dir = '/media/Data-B/data/main_data/train_cat_new/turning_around_train'
val_dir = '/media/Data-B/data/main_data/val_cat_new/turning_around_val'

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# create a list of image paths use tqdm here
image_paths = []
for root, dirs, files in os.walk(input_dir):
    for file in tqdm(files):
        if file.endswith('.jpg'):
            img_path = os.path.join(root, file)
            image_paths.append(img_path)

# split the list of image paths into train and val sets
train_paths, val_paths = train_test_split(image_paths, test_size=0.1)

# copy the train images to output_train_dir
for img_path in train_paths:
    shutil.copy(img_path, train_dir)

# copy the val images to output_val_dir
for img_path in val_paths:
    shutil.copy(img_path, val_dir)
