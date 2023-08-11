import csv
import os
import shutil
from tqdm import tqdm
import argparse

def main(input_dir, output_dir, csv_file, target_category):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load image categories from CSV file
    image_categories = {}
    with open(csv_file) as csvfile:
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
                if category == target_category:
                    category_dir = os.path.join(output_dir, category)
                    if not os.path.exists(category_dir):
                        os.makedirs(category_dir)
                    shutil.copy(image_path, os.path.join(category_dir, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize images based on a specific category using a CSV file.")
    parser.add_argument("--input_dir", required=True,  help="Path to the input directory containing image files.")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory where images will be organized.")
    parser.add_argument("--csv_file", required=True, help="Path to the CSV file containing image categories.")
    parser.add_argument("--target_category", required=True, help="The target category to organize images for.")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.csv_file, args.target_category)