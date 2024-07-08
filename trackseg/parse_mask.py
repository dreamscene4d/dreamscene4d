import os
import cv2
import numpy as np
import shutil
import argparse

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def separate_masks(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    create_directory(output_folder)
    
    # Get the list of all mask files in the input folder
    mask_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    unique_colors = None

    for index, mask_file in enumerate(mask_files):
        # Read the mask image
        mask_path = os.path.join(input_folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        
        if index == 0:
            # Get unique colors in the mask
            unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
            
            # Skip the background color (assuming background is black)
            unique_colors = [color for color in unique_colors if not np.array_equal(color, [0, 0, 0])]
        
        
        for color in unique_colors:
            # Create a binary mask for each color
            binary_mask = cv2.inRange(mask, color, color)
            
            # Create a folder for the current color/object
            color_folder = os.path.join(output_folder, f"object_{color[0]}_{color[1]}_{color[2]}")
            create_directory(color_folder)
            
            # Save the binary mask
            binary_mask_filename = os.path.join(color_folder, mask_file)
            cv2.imwrite(binary_mask_filename, binary_mask)

def rename_folders(base_path):
    # List all folders in the base path
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    # Sort the folders
    folders.sort()
    
    # Rename the folders
    for index, folder in enumerate(folders, start=1):
        new_name = f"{index:03}"  # Format index as three-digit number
        old_path = os.path.join(base_path, folder)
        new_path = os.path.join(base_path, new_name)
        
        # Rename the folder
        os.rename(old_path, new_path)
        print(f"Renamed '{folder}' to '{new_name}'")

def create_class_output_folders(output_folder, class_output_folder):
    # List all '00x' folders in the output_folder
    folders = [f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))]
    # Sort the folders to maintain order
    folders.sort()
    
    # Create the class_output_folder
    os.makedirs(class_output_folder, exist_ok=True)
    
    # Create '00x' folders and 'class_label.txt' files in class_output_folder
    for folder in folders:
        # Create the '00x' folder in class_output_folder
        folder_path = os.path.join(class_output_folder, folder)
        os.makedirs(folder_path, exist_ok=True)
        
        # Create 'class_label.txt' file with the word 'cup'
        class_label_file = os.path.join(folder_path, "class_label.txt")
        with open(class_label_file, 'w') as f:
            f.write("cup")
        
        print(f"Created folder '{folder}' and class_label.txt in '{class_output_folder}'")

def create_folder_structure(base_path, folder_structure):
    # Combine the base path with the folder structure
    full_path = os.path.join(base_path, folder_structure)
    
    # Create the directories, if they don't exist
    os.makedirs(full_path, exist_ok=True)
    
    print(f"Folder structure '{full_path}' created successfully!")

def main(roof_f, names):
    # Usage
    input_folder = roof_f + "Annotations/"
    output_folder = roof_f + "OriMasks/"
    
    case_name = names + '/'
    separate_masks(input_folder, output_folder + case_name)
    
    # Usage
    rename_folders(output_folder + case_name)
    
    class_output_folder = output_folder.replace('OriMasks', 'ClassLabels') + case_name
    
    # Usage
    create_class_output_folders(output_folder, class_output_folder)
    
    # Example usage
    folder_structure = 'OriImages/' + names + '/001'
    create_folder_structure(roof_f, folder_structure)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the roof_f folder and names variable")
    parser.add_argument('roof_f', type=str, help="The path to the roof_f folder")
    parser.add_argument('names', type=str, help="The names variable")

    args = parser.parse_args()
    main(args.roof_f, args.names)
