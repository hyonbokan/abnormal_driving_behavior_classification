import csv
import os
import shutil
from tqdm import tqdm


image_categories = {}

with open('../val_img_name_and_action.csv') as csvfile:
    reader = csv.reader(csvfile)
    # skip the header row
    next(reader) 
    for row in reader:
        image_name, category = row
        image_categories[image_name] = category
# print(image_categories)


input_dir = '/media/Data-B/org_data_abnormal_driving/2.Validation/원천데이터/abnormal_230303_add'
output_dir = '/home/dnlab/Data-B/data/main_data/test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for root, dirs, files in os.walk(input_dir):
    for file in tqdm(files):
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            # print(image_path)
            image_name = file.split('.')[0] + '.jpg'
            # print(image_name)
            category = image_categories.get(image_name, 'other')
            # print(category)
            category_dir = os.path.join(output_dir, category)
            # create a new directory for the category if it doesn't exist
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
            # copy the image to the category directory
            shutil.copy(image_path, os.path.join(category_dir, file))