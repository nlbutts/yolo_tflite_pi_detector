import os
import json
import glob
import shutil
import argparse
from PIL import Image
from sklearn.model_selection import train_test_split

def convert_labelme_to_yolo(labelme_json_dir, output_dir, category_mapping, train_ratio):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare train and val directories
    train_image_dir = os.path.join(output_dir, 'train', 'images')
    val_image_dir = os.path.join(output_dir, 'val', 'images')
    train_label_dir = os.path.join(output_dir, 'train', 'labels')
    val_label_dir = os.path.join(output_dir, 'val', 'labels')

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    labelme_json_files = glob.glob(os.path.join(labelme_json_dir, '*.json'))

    # Split data into train and val sets
    train_files, val_files = train_test_split(labelme_json_files, train_size=train_ratio, random_state=42)

    for labelme_json_file in labelme_json_files:
        with open(labelme_json_file, 'r') as f:
            data = json.load(f)

        image_path = data['imagePath']
        full_path = labelme_json_dir + '/' + image_path
        image = Image.open(full_path)
        width, height = image.size

        # Determine if the file is for train or val
        if labelme_json_file in train_files:
            image_output_dir = train_image_dir
            label_output_dir = train_label_dir
        else:
            image_output_dir = val_image_dir
            label_output_dir = val_label_dir

        # Copy image to the corresponding directory
        shutil.copy(full_path, os.path.join(image_output_dir, os.path.basename(image_path)))

        # Create corresponding YOLO format .txt file
        txt_file_name = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
        txt_file_path = os.path.join(label_output_dir, txt_file_name)

        with open(txt_file_path, 'w') as txt_file:
            for shape in data['shapes']:
                label = shape['label']

                print(label)
                print(category_mapping)
                if label not in category_mapping:
                    continue

                category_id = category_mapping[label]
                points = shape['points']

                # Calculate YOLO format coordinates
                min_x = min([p[0] for p in points])
                min_y = min([p[1] for p in points])
                max_x = max([p[0] for p in points])
                max_y = max([p[1] for p in points])

                bbox_width = max_x - min_x
                bbox_height = max_y - min_y

                # YOLO format requires normalized center coordinates
                center_x = (min_x + bbox_width / 2) / width
                center_y = (min_y + bbox_height / 2) / height
                norm_width = bbox_width / width
                norm_height = bbox_height / height

                txt_file.write(f"{category_id} {center_x} {center_y} {norm_width} {norm_height}\n")

def main():
    parser = argparse.ArgumentParser(description="Convert LabelMe JSON annotations to YOLO format and split into train/val sets.")
    parser.add_argument('labelme_json_dir', type=str, help="Directory containing LabelMe JSON files.")
    parser.add_argument('output_dir', type=str, help="Output directory for YOLO formatted data.")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="Ratio of training data (default: 0.8)")
    parser.add_argument('--category_mapping', type=str, required=True, help="Path to JSON file with category mapping.")

    args = parser.parse_args()

    # Load category mapping
    with open(args.category_mapping, 'r') as f:
        category_mapping = json.load(f)

    convert_labelme_to_yolo(args.labelme_json_dir, args.output_dir, category_mapping, args.train_ratio)

if __name__ == "__main__":
    main()
