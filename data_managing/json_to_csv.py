import os
import shutil
import json
import csv
from tqdm import tqdm

# specify the input directory containing the JSON files
input_dir = '/media/Data-B/org_data_abnormal_driving/2.Validation/라벨링데이터/abnormal_230303_add'

# specify the output CSV file
output_file = '/media/Data-B/my_research/abnormal_driving_FL/val_img_name_and_action.csv'

# create a list to store the image names and emotions
image_action_dir = {}

# loop through each JSON file in the directory and subdirectories
for root, dirs, files in tqdm(os.walk(input_dir)):
    for file in files:
        if file.endswith('.json'):
            # read the JSON file
            with open(os.path.join(root, file)) as f:
                file_data = json.load(f)
                for i in file_data['scene']['data']:
                    img_name = i['img_name']
                    # print(img_name)
                    # emotion = i['occupant']
                    action = i['occupant'][0]['action']
                    # print(action)
                    image_action_dir[img_name] = action

# print(image_action_dir)

# save the dictionary to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['img_name', 'action'])
    for img_name, action in image_action_dir.items():
        writer.writerow([img_name, action])