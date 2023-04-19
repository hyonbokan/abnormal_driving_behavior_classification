import csv
import os
import shutil
from tqdm import tqdm

input_dir = '/home/dnlab/Data-B/data/main_data/train_ppr'
output_dir = '/home/dnlab/Data-B/data/main_data/train_ppr_cat/missing'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load image categories from CSV file
image_categories = {}
with open('../train_img_name_and_action.csv') as csvfile:
    reader = csv.reader(csvfile)
    # skip the header row
    next(reader) 
    for row in reader:
        image_name, category = row
        image_categories[image_name] = category

for root, dirs, files in os.walk(input_dir):
    for file in tqdm(files):
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            image_name = file.split('.')[0] + '.jpg'
            category = image_categories.get(image_name, 'other')
            if category == '운전자를향해발을뻗다':  # modify this line to match the desired category
                category_dir = os.path.join(output_dir, category)
                # create a new directory for the category if it doesn't exist
                if not os.path.exists(category_dir):
                    os.makedirs(category_dir)
                # copy the image to the category directory
                shutil.copy(image_path, os.path.join(category_dir, file))
