import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

def main(input_dir, train_dir, val_dir):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                image_paths.append(img_path)

    train_paths, val_paths = train_test_split(image_paths, test_size=0.2)

    for img_path in train_paths:
        shutil.copy(img_path, train_dir)

    for img_path in val_paths:
        shutil.copy(img_path, val_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split images into train and validation sets.")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing image files.")
    parser.add_argument("--train_dir", required=True, help="Path to the output directory for train images.")
    parser.add_argument("--val_dir", required=True, help="Path to the output directory for validation images.")
    args = parser.parse_args()

    main(args.input_dir, args.train_dir, args.val_dir)
