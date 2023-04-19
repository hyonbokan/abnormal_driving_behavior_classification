import os
import shutil
import json
import csv
from tqdm import tqdm


# specify the input directory containing the JSON files
input_dir = '/media/Data-B/org_data_abnormal_driving/1.Training/라벨링데이터/abnormal_230303_add'

# specify the output CSV file
output_file = '/home/dnlab/Data-B/my_research/abnormal_driving_FL/train_img_name_and_action.csv'

# create a list to store the image names, actions and IDs
img_data = []

# loop through each JSON file in the directory and subdirectories
for root, dirs, files in tqdm(os.walk(input_dir)):
    for file in files:
        if file.endswith('.json'):
            # read the JSON file
            with open(os.path.join(root, file)) as f:
                file_data = json.load(f)
                for i in file_data['scene']['data']:
                    img_name = i['img_name']
                    occupant_id = i['occupant'][0]['occupant_id']
                    action = i['occupant'][0]['action']
                    img_data.append([occupant_id, action, img_name])

# save the data to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'action', 'img_name'])
    writer.writerows(img_data)
