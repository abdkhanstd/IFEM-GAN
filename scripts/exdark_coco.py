import os
import json
from PIL import Image
from tqdm import tqdm
from shutil import copyfile

# Define the list of labels
labels = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']

# Define a label map to map label names to integers
label_map = {i: label for i, label in enumerate(labels)}

def convert_to_coco_format(dataset_root, output_dir, val_txt_path):
    """
    Converts the Exdark dataset into COCO object detection annotation style.

    Args:
        dataset_root (str): The root directory of the Exdark dataset containing "labels" and "images" folders.
        output_dir (str): The directory where the COCO format JSON files will be saved.
        val_txt_path (str): The path to the val.txt file containing the list of filenames for the test set.

    Note:
        This function assumes that the Exdark dataset is already standardized with label files in YOLO format.
    """
    # Read the list of test filenames from val.txt
    with open(val_txt_path, 'r') as val_file:
        test_filenames = val_file.read().splitlines()

    # Create the COCO format dictionary
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": label} for i, label in label_map.items()]
    }

    # Initialize image and annotation IDs
    image_id = 1
    annotation_id = 1

    images_directory = os.path.join(dataset_root, 'images')
    labels_directory = os.path.join(dataset_root, 'labels')

    # Define train and test output directories
    train_output_dir = os.path.join(output_dir, 'train')
    test_output_dir = os.path.join(output_dir, 'test')

    for label in labels:
        label_directory = os.path.join(labels_directory, label)
        filenames = os.listdir(label_directory)

        for filename in tqdm(filenames, desc=f'Converting {label} annotations'):
            image_filename = os.path.splitext(filename)[0] + '.jpg'
            image_path = os.path.join(images_directory, label, image_filename)
            annotation_path = os.path.join(label_directory, filename)

            # Check if the image file exists
            if not os.path.exists(image_path):
                continue

            # Load the image to get its dimensions
            image = Image.open(image_path)
            width, height = image.size

            # Create an image entry in the COCO data
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height,
            })

            # Read YOLO format annotations
            with open(annotation_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                left, top, bbox_width, bbox_height = map(float, parts[1:5])

                # Calculate bounding box coordinates
                x_min = int(left)
                y_min = int(top)
                x_max = int(left + bbox_width)
                y_max = int(top + bbox_height)

                # Create an annotation entry in the COCO data
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0,
                })

                annotation_id += 1

            # Copy the image to the appropriate train or test directory
            if filename in test_filenames:
                os.makedirs(os.path.join(test_output_dir, label), exist_ok=True)
                copyfile(image_path, os.path.join(test_output_dir, label, image_filename))
            else:
                os.makedirs(os.path.join(train_output_dir, label), exist_ok=True)
                copyfile(image_path, os.path.join(train_output_dir, label, image_filename))

            image_id += 1

    # Save the COCO data in JSON format
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, 'annotations.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_data, json_file)

if __name__ == '__main__':
    dataset_root = '../datasets/ExDarkOriginalForm'  # Update with your dataset root path
    output_dir = '../datasets/ExDark3'  # Update with your desired output directory
    import abdutils as abd
    abd.CreateFolder(output_dir)
    val_txt_path = '../datasets/ExDarkOriginalForm/val.txt'  # Update with the path to val.txt
    convert_to_coco_format(dataset_root, output_dir, val_txt_path)
