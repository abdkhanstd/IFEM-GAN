import os
from PIL import Image
import argparse
import os
import scripts.abdutils as abd
from tqdm import tqdm


# Define the list of labels
labels = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']

# Define a label map to map label names to integers
label_map = {i: label for i, label in enumerate(labels)}


def StandardiseDataSet(dataset_root):
    """
    Standardizes file extensions and renames label files in the specified dataset directory.

    Args:
        dataset_root (str): The root directory of the dataset containing "labels" and "images" folders.

    Note:
        This function assumes that you want to standardize file extensions in the "images" directory
        and its subdirectories to '.jpg', and rename label files with problematic names in the "labels" directory.
    """
    # Convert the provided dataset root path to an absolute path
    dataset_root = os.path.abspath(dataset_root)

    print("Dataset Root:", dataset_root)

    # Define a mapping of file extensions to a preferred extension
    extension_mapping = {
        '.jpeg': '.jpg',
        '.JPG': '.jpg',
        '.JPEG': '.jpg',
        '.png': '.jpg',
        '.PNG': '.jpg',
    }

    # Standardize file extensions in the "images" directory and its subdirectories
    images_directory = os.path.join(dataset_root, 'images')
    for root, dirs, files in tqdm(os.walk(images_directory), desc='Processing Directories 12/ ', unit=' Classes'):
        for file in tqdm(files, desc='Standardizing Images', unit='image', leave=False):
            file_name, file_ext = os.path.splitext(file)

            # Check if the file extension is in the mapping
            if file_ext in extension_mapping:
                # Rename the file with the preferred extension
                new_file_name = file_name + extension_mapping[file_ext]
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, new_file_name)

                # Rename the file                
                #abd.Rename(old_file_path, new_file_path)
                
                png_image = Image.open(old_file_path)

                # Convert it to JPEG format
                jpeg_image = png_image.convert('RGB')

                # Save the converted image as a JPEG
                jpeg_image.save(new_file_path)

                # Close the image files
                png_image.close()
                jpeg_image.close()                
                
                
                
                #handel the label as well
                old_file_path = os.path.join(root, file)+'.txt'
                old_file_path = old_file_path.replace("images","labels")
                new_file_path = os.path.join(root, new_file_name)+'.txt'
                new_file_path = new_file_path.replace("images","labels")
                # Rename the file
                
                if os.path.isfile(old_file_path):
                    abd.Rename(old_file_path, new_file_path,verbose=False)
                else:
                    # Had to do this because with incosistant file names in the dataset
                    old_file_path=old_file_path.replace("JPEG","jpg")
                    old_file_path=old_file_path.replace("JPG","jpg")
                    abd.Rename(old_file_path, new_file_path,verbose=False)


def SearchFileInText(file_name, text_file_path):
    try:
        while True:  # Loop to keep reading lines until we reach the end of the file
            line = abd.ReadFile(text_file_path)
            if line is None:
                break  # End of the file

            #print(line)  # Print each line, as per the original function

            if file_name in line:
                return True  # File name found in the text file

        return False  # File name not found in the text file

    except FileNotFoundError:
        print(f"Error: The text file '{text_file_path}' does not exist.")
        return False


def check(center_x, center_y, yolo_width, yolo_height):
    """
    Check if any of the normalized YOLO values exceed 1.

    Returns:
        bool: True if any normalized value exceeds 1, False otherwise.
    """
    if center_x > 1.0 or center_y > 1.0 or yolo_width > 1.0 or yolo_height > 1.0:
        return True
    else:
        return False

def ExDark2Yolo8(txts_dir: str, imgs_dir: str, output_dir: str):
  
    # Create output directories for images and labels
    abd.CreateFolder(os.path.join(output_dir, 'images', 'test'))
    abd.CreateFolder(os.path.join(output_dir, 'images', 'train'))
    abd.CreateFolder(os.path.join(output_dir, 'labels', 'test'))
    abd.CreateFolder(os.path.join(output_dir, 'labels', 'train'))    
        

    ct=0
    ctr=0
    
    # Calculate the total number of iterations
    total_iterations = sum(len(os.listdir(os.path.join(txts_dir, label))) for label in labels)
    
    with tqdm(total=total_iterations, desc='Processing labels and files') as pbar:    
        for label in labels:
            
            filenames = os.listdir(os.path.join(txts_dir, label))
            cur_idx = 0
            files_num = len(filenames)

            for filename in filenames:
                pbar.update(1)  # Update the progress bar for each file processed
                
                pbar.set_description(f'Processing label: {label}/{filename}')

                cur_idx += 1
                file_ext=os.path.splitext(filename)[1]
                filename_no_ext = '.'.join(filename.split('.')[:-2])
                
                text_file_path = "datasets/ExDark_stock/val.txt"   
                found =  SearchFileInText(filename_no_ext, text_file_path)

                if found:
                    set_type = 'test'
                    ct=ct+1
                else:
                    set_type = 'train'
                    ctr=ctr+1
                
                

                output_label_path = os.path.join(output_dir, 'labels', set_type, filename_no_ext + '.txt')
                #yolo_output_file = open(output_label_path, 'a')

                name_split = filename.split('.')
                img_path = os.path.join(imgs_dir, label, '.'.join(filename.split('.')[:-1]))

    
                img=abd.ReadImage(img_path)
                file_name = os.path.basename(img_path)

                # Update the output image path
                output_img_path = os.path.join(output_dir, 'images', set_type, file_name)
                output_img_path = output_img_path.replace("//", "/")
                output_img_path = output_img_path.replace("png", "jpg")
                output_img_path = output_img_path.replace("PNG", "jpg")
                output_img_path = output_img_path.replace("JPG", "jpg")
                output_img_path = output_img_path.replace("JPEG", "jpg")
                output_img_path = output_img_path.replace("jpeg", "jpg")

                # Convert the image to RGB and save it
                img = img.convert("RGB")
                img.save(output_img_path)

                width_image, height_image = img.size
                OriginalLableFileName=os.path.join(txts_dir, label, filename)
                
                # Ignore First Line
                line=abd.ReadFile(OriginalLableFileName)
                line=abd.ReadFile(OriginalLableFileName)
                
                
                
                yolo_annotations = []

                while line is not None:
                    parts = line.strip().split()
                
                    # Extract object properties from the annotation file
                    object_class = [key for key, value in label_map.items() if value == parts[0]][0]
                    left = int(parts[1])
                    top = int(parts[2])
                    width = int(parts[3])
                    height = int(parts[4])

                    # Calculate YOLO-style coordinates
                    center_x = (left + (width / 2)) / width_image  # Normalize by image width 
                    center_y = (top + (height / 2)) / height_image  # Normalize by image height
                    yolo_width = width / width_image  # Normalize by image width
                    yolo_height = height / height_image  # Normalize by image height
                    
                    if check(center_x, center_y, yolo_width, yolo_height):
                        print("Warning: Normalized values exceed 1. Please check your annotations.")
                        exit(0)
               

                    # Append the YOLO-style annotation to the list
                    yolo_annotation = f"{object_class} {center_x:.6f} {center_y:.6f} {yolo_width:.6f} {yolo_height:.6f}\n"
                    yolo_annotations.append(yolo_annotation)

                    #Read the next line
                    line=abd.ReadFile(OriginalLableFileName)

                yolo_annotations = "".join(yolo_annotations)

                # Write the YOLO annotations to the label file            
                abd.WriteFile(output_label_path,yolo_annotations)
                
        print("Converted Train: ", ctr, "Test ", ct, "images and label files")


def ProcessExDarkForYolo():   
    # This willsplit according to test train used in previous paper and adjust the coordinateds according to YOLO8 style
    abd.Delete('datasets/ExDark_stock')
    abd.Copy('datasets/ExDarkOriginalForm','datasets/ExDark_stock')
    
    # Need to standardise directory as theres problems with DS    
    StandardiseDataSet('datasets/ExDark_stock')
        
    output_dir='datasets/ExDark'
    # Removing old target directory
    abd.Delete(output_dir)
        
    annotations_dir='datasets/ExDark_stock/labels'
    images_dir='datasets/ExDark_stock/images'

    ExDark2Yolo8(annotations_dir, images_dir, output_dir)
    
    # Delete back the temp folder
    abd.Delete('datasets/ExDark_stock')


if __name__ == '__main__':
   
    ProcessExDarkForYolo()

