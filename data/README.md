# Data Processing Scripts

The directory contains Python scripts for data processing tasks. Each script is designed to perform specific data manipulation or organization tasks. Below is a brief overview of each script:

## Script 1: `parse_json.py`

Converts JSON data to CSV format. It reads JSON files from a specified input directory and outputs the data in CSV format to a specified output CSV file. The script utilizes the `argparse` library for command-line argument parsing.

```bash
python json_to_csv_converter.py --input_dir /path/to/input_directory --output_csv /path/to/output_csv_file.csv
```

## Script 2: train_val_split.py
Organizes a collection of image files into train and validation sets. It takes an input directory containing image files and splits them into train and validation directories. The split ratio can be adjusted by modifying the script. The script utilizes the train_test_split function from the sklearn library.

```bash
python image_data_organizer.py --input_dir /path/to/input_directory --train_dir /path/to/train_directory --val_dir /path/to/validation_directory
```

## Script 3: image_truncated_checker.py
This script checks for and removes truncated or corrupted image files in a specified directory. It verifies each image using the PIL library and removes any files that are not valid images.

```bash
python image_truncated_checker.py --input_dir /path/to/input_directory
```
## Script 4: image_category_organizer.py
This script organizes image files into category-specific directories based on data from a CSV file. It reads a CSV file containing image names and corresponding categories and copies the images to category directories accordingly.

```bash
python image_category_organizer.py --input_dir /path/to/input_directory --output_dir /path/to/output_directory --csv_file /path/to/csv_file.csv
```

## Script 5: image_category_organizer_v2.py
An enhanced version of the previous script, this script also allows to organize images belonging to a specific category, effectively filtering out other categories.

```bash
python image_category_organizer.py --input_dir /path/to/input_directory --output_dir /path/to/output_directory --csv_file /path/to/csv_file.csv --target_category desired_category
```


## Script 6: copy_move_img.py
The script is designed to address the issue of data imbalance in a dataset by copying a random selection of image files from a source directory to a destination directory. This script is particularly useful when dealing with image datasets where certain categories have a significantly larger number of images compared to others, leading to an uneven distribution of data.

```bash
python copy_move_img.py --src_dir <path_to_source_directory> --dst_dir <path_to_destination_directory>
```
