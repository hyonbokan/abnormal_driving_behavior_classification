import os
import shutil
import json
import csv
from tqdm import tqdm
import argparse

def main(input_dir, output_file):
    image_action_dir = {}
    
    for root, dirs, files in tqdm(os.walk(input_dir)):
        for file in files:
            if file.endswith('.json'):
                # read the JSON file
                with open(os.path.join(root, file)) as f:
                    file_data = json.load(f)
                    for i in file_data['scene']['data']:
                        img_name = i['img_name']
                        action = i['occupant'][0]['action']
                        image_action_dir[img_name] = action

    # save the dictionary to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['img_name', 'action'])
        for img_name, action in image_action_dir.items():
            writer.writerow([img_name, action])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parses original JSON data and converts it into CSV.")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing JSON files.")
    parser.add_argument("--output_file", required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    main(args.input_dir, args.output_file)
