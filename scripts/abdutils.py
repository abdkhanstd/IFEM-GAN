__version__ = '0.1'
# https://github.com/abdkhanstd/abdutils
import os
import shutil
import warnings
import cv2
import numpy as np
from PIL import ImageFilter, Image, ImageEnhance
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import inspect
import sys
import random
import glob
import GPUtil
import threading
import time
import signal
import os
import subprocess
import threading
import platform

import threading
import time
import psutil
import GPUtil
import shutil

# Function to get CPU, GPU, and Disk usage
def GetSystemUsage():
    cpu_usage = psutil.cpu_percent(interval=1)
    gpus = GPUtil.getGPUs()
    gpu_usages = [gpu.load * 100 for gpu in gpus] if gpus else ['N/A']
    gpu_memory = [gpu.memoryUtil * 100 for gpu in gpus] if gpus else ['N/A']
    disk_usage = psutil.disk_usage('/').percent
    return cpu_usage, gpu_usages, gpu_memory, disk_usage

# Function to get the current console height
def GetConsoleHeight():
    return shutil.get_terminal_size((80, 20)).lines

# Function to update the system usage display
def ShowSystemUsage():
    while True:
        console_height = GetConsoleHeight()
        cpu, gpu_usages, gpu_memory, disk = GetSystemUsage()
        gpu_usage_str = ' | '.join([f"GPU {i}: {usage:.2f}%" for i, usage in enumerate(gpu_usages)])
        gpu_memory_str = ' | '.join([f"Memory {i}: {memory:.2f}%" for i, memory in enumerate(gpu_memory)])
        system_info = f"CPU: {cpu}% | {gpu_usage_str} | {gpu_memory_str} | Disk: {disk}%"

        # Move cursor to the bottom line and print the system info
        print(f"\033[{console_height};0H\033[K{system_info}", end='', flush=True)

        time.sleep(1)

# Function to add a new message above the system usage
def add_message(message):
    console_height = GetConsoleHeight()
    # Clear and move to the beginning of the line just above the bottom
    print(f"\033[{console_height-1};0H\033[K", end='')
    # Move up to make space for the new message
    print(f"\033[1A{message}")

# Start the system usage display in a separate thread
def ShowUsage():
    threading.Thread(target=ShowSystemUsage, daemon=True).start()
    
def ClearScreen():
    # Check if the operating system is Windows
    if platform.system().lower() == "windows":
        os.system('cls')
    else:
        # For Unix and MacOS
        os.system('clear')

# Global flag to indicate whether the exit handler has been executed
exit_handler_executed = False

# Function to capture Ctrl+C and perform actions
def ExitHandler(signum, frame):
    global exit_handler_executed

    if not exit_handler_executed:
        exit_handler_executed=True
        pid = os.getpid()
        msg = f"[🛑 Interrupted] Stopping your code and Killing PID {pid}"
        print(msg)
        
        # Perform your desired actions here
        os.kill(os.getpid(), signal.SIGKILL)  # For example, terminate the process

        # Run the 'kill -9' command to terminate the process
        try:
            subprocess.run(["kill", "-9", str(pid)], check=True)
            
        except subprocess.CalledProcessError as e:
            msg = f"[🛑 Interrupt Error] {e}"
            print(msg)

        # Set the flag to indicate that the handler has been executed
        exit_handler_executed = True
            
# Function to start Ctrl+C capture as a side daemon
def LookForKeys():
    # Set up a handler for Ctrl+C (SIGINT)
            
    signal.signal(signal.SIGINT, ExitHandler)
    signal.signal(signal.SIGTSTP, ExitHandler)



def SelectGPU():
    
    caller_filename, caller_line=get_caller_info()
    try:
        # Get a list of available GPUs
        gpus = GPUtil.getGPUs()

        if not gpus:
            print("No GPUs found.")
            return None

        # Sort the list of GPUs by memory usage (ascending order)
        gpus.sort(key=lambda gpu: gpu.memoryFree, reverse=True)

        # Select the GPU with the least used memory (the first GPU in the sorted list)
        selected_gpu = gpus[0]

        msg=(f"🖥️🖥️ Selected GPU ID: {selected_gpu.id} {selected_gpu.name} (Free Memory: {selected_gpu.memoryFree} MB)")
        ShowInfo(msg,caller_filename,caller_line)
        
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu.id)

        return selected_gpu, selected_gpu.id

    except Exception as e:
        print(f"Error: {e}")
        return None

def HandleError(msg, caller_filename, caller_line):    
    print(f"[🚫 Error: {caller_filename}, line {caller_line}] " + msg)
    exit(0)
    
def ShowInfo(msg, caller_filename, caller_line):    
    print(f"[📌 Info: {caller_filename}, line {caller_line}] " + msg)
    
def ShowWarning(msg, caller_filename, caller_line):    
    print(f"[⚠️ Warning: {caller_filename}, line {caller_line}] " + msg)    

def check_required_args():
    # Get the calling function's frame
    caller_frame = inspect.currentframe().f_back
    # Get the calling function's arguments
    caller_args = inspect.getargvalues(caller_frame).locals

    # Extract argument names of the calling function
    func_arg_names = inspect.getfullargspec(caller_frame.f_globals[caller_frame.f_code.co_name]).args

    missing_args = [arg_name for arg_name in func_arg_names if caller_args[arg_name] is None]

    if missing_args:
        caller_frame = sys._getframe(2)  # Get the caller's frame (1 level up in the call stack)
        caller_line = caller_frame.f_lineno  # Get the caller's line number
        caller_filename = caller_frame.f_globals.get('__file__')  # Get the caller's filename

        msg = f"The following input(s) /argument(s) are missing: {', '.join(missing_args)}"
        HandleError(msg, caller_filename, caller_line)
        

       
def get_caller_info():    
    caller_frame = sys._getframe(2)
    caller_line = caller_frame.f_lineno
    caller_filename = caller_frame.f_globals.get('__file__')
    return caller_filename, caller_line



def PrintObject(obj):    
    # Print all attributes of the results object
    for attr in dir(obj):
        if not attr.startswith('_'):
            print(f"{attr}: {getattr(obj, attr)}")



def ReadDirectoryContents(path_pattern=None, verbose=True):
    """
    Reads the contents of a directory based on the provided pattern and returns a list of matched items.

    Args:
        path_pattern (str): The path pattern to match files and directories. 
                            For example: '/home/tt/*.jpg' or '/home/tt/*.*' or '/home/tt/'
        verbose (bool): Whether to display verbose messages. Defaults to True.

    Returns:
        list: A list of matched items based on the provided pattern.
    """
    
    check_required_args()
    caller_filename, caller_line=get_caller_info()
        
    try:
        matched_items = glob.glob(path_pattern)
        if verbose:
            msg=(f"Found {len(matched_items)} items matching the pattern '{path_pattern}'.")

        return matched_items

    except Exception as e:
        HandleError(str(e), caller_filename, caller_line)
        return []

def GetFileNameFromPath(path):
    """
    Split a path into its folder and filename components.

    Args:
        path (str): The path to be split.

    Returns:
        tuple: A tuple containing the folder and filename components.
    """
    folder = os.path.dirname(path)
    filename = os.path.basename(path)
    return folder, filename

def Copy(src_path=None, dest_path=None, verbose=True):
    """
    Copies files from the source pattern to the destination path.

    Args:
        src_path (str): The source file pattern.
        dest_path (str): The destination path.
        verbose (bool): Whether to display verbose messages. Defaults to True.
    """
    check_required_args()
    caller_filename, caller_line = get_caller_info()

    try:
        # Create the destination directory if it doesn't exist
        os.makedirs(dest_path, exist_ok=True)

        # Check if src_path is a wildcard pattern
        if '*' in src_path:
            # Use glob to expand the source pattern and get a list of matching files/folders
            matched_paths = glob.glob(src_path)

            if not matched_paths:
                msg = f"No files/folders found matching the pattern '{src_path}'."
                raise FileNotFoundError(msg)

            # Copy each matched file/folder to the destination directory
            for src_item in matched_paths:
                dest_item = os.path.join(dest_path, os.path.basename(src_item))
                if os.path.isdir(src_item):
                    # Copy the contents of the folder directly into the destination folder
                    for item in os.listdir(src_item):
                        src_subitem = os.path.join(src_item, item)
                        dest_subitem = os.path.join(dest_path, item)
                        if os.path.isdir(src_subitem):
                            shutil.copytree(src_subitem, dest_subitem)
                        else:
                            shutil.copy(src_subitem, dest_subitem)
                else:
                    shutil.copy(src_item, dest_item)
            if verbose:
                msg=(f"Copied {len(matched_paths)} items to '{dest_path}'.")
                ShowInfo(msg,caller_filename,caller_line)

        else:
            # Check if src_path is a file or folder
            if os.path.isfile(src_path):
                dest_item = os.path.join(dest_path, os.path.basename(src_path))
                shutil.copy(src_path, dest_item)
                if verbose:
                    msg=(f"Copied file '{src_path}' to '{dest_item}'.")
                    ShowInfo(msg,caller_filename,caller_line)

            elif os.path.isdir(src_path):
                # Copy the contents of the folder directly into the destination folder
                for item in os.listdir(src_path):
                    src_subitem = os.path.join(src_path, item)
                    dest_subitem = os.path.join(dest_path, item)
                    if os.path.isdir(src_subitem):
                        shutil.copytree(src_subitem, dest_subitem)
                    else:
                        shutil.copy(src_subitem, dest_subitem)
                if verbose:
                    msg=(f"Copied folder '{src_path}' to '{dest_path}'.")
                    ShowInfo(msg,caller_filename,caller_line)
            else:
                msg = f"'{src_path}' does not exist or is not a valid file/folder."
                HandleError(msg, caller_filename, caller_line)                
                raise FileNotFoundError(msg)

    except Exception as e:
        HandleError(str(e), caller_filename, caller_line)


def Move(src_path=None, dest_path=None, verbose=True):
    """
    Moves files and folders from the source pattern to the destination path.

    Args:
        src_path (str): The source file or folder path.
        dest_path (str): The destination path.
        verbose (bool): Whether to display verbose messages. Defaults to True.
    """
    
    check_required_args()
    caller_filename, caller_line = get_caller_info()    
    try:
        # Check if src_path is a wildcard pattern
        if '*' in src_path:
            # Use glob to expand the source pattern and get a list of matching files/folders
            matched_paths = glob.glob(src_path)

            if not matched_paths:
                msg = f"No files/folders found matching the pattern '{src_path}'."
                HandleError(msg, caller_filename, caller_line)
                raise FileNotFoundError(msg)

            # Move each matched file/folder to the destination directory
            for src_item in matched_paths:
                dest_item = os.path.join(dest_path, os.path.basename(src_item))
                shutil.move(src_item, dest_item)
                if verbose:
                    msg=(f"Moved {os.path.basename(src_item)} to '{dest_item}'.")
                    ShowInfo(msg, caller_filename, caller_line)
        else:
            # Check if src_path is a file or folder
            if os.path.isfile(src_path):
                # If src_path is a file, construct the destination file path
                dest_file_path = os.path.join(dest_path, os.path.basename(src_path))
                shutil.move(src_path, dest_file_path)
                if verbose:
                    msg=(f"Moved file '{src_path}' to '{dest_file_path}'.")
                    ShowInfo(msg, caller_filename, caller_line)
            elif os.path.isdir(src_path):
                # If src_path is a directory, construct the destination directory path
                dest_dir_path = os.path.join(dest_path, os.path.basename(src_path))
                shutil.move(src_path, dest_dir_path)
                if verbose:
                    msg=(f"Moved folder '{src_path}' to '{dest_dir_path}'.")
                    ShowInfo(msg, caller_filename, caller_line)
            else:
                msg = f"'{src_path}' does not exist or is not a valid file/folder."
                HandleError(msg, caller_filename, caller_line)
                raise FileNotFoundError(msg)

    except Exception as e:
        msg = f"[Error] {str(e)}"
        HandleError(msg, caller_filename, caller_line)




def Delete(path=None, verbose=True):
    """
    Deletes files or folders based on the given path. Supports wildcard patterns.

    Args:
        path (str): The path to the file or folder to be deleted. Supports wildcard patterns.
        verbose (bool): Whether to display verbose messages. Defaults to True.
    """
    check_required_args()
    caller_filename, caller_line = get_caller_info()

    try:
        # Use glob to expand the path and get a list of matching files and folders
        matched_paths = glob.glob(path, recursive=True)

        if not matched_paths:
            if verbose:
                msg = f"No files/folders found matching the pattern. Skipping Deletion'{path}'."
            return

        # Iterate over matched paths and delete them
        for matched_path in matched_paths:
            try:
                if os.path.isfile(matched_path):
                    # Delete the file
                    os.remove(matched_path)
                    if verbose:
                        msg=(f"Deleted file '{matched_path}'.")
                elif os.path.isdir(matched_path):
                    # Delete the folder and its contents
                    shutil.rmtree(matched_path)
                    if verbose:
                        msg=(f"Deleted folder '{matched_path}' and its contents.")
                        ShowInfo(msg, caller_filename, caller_line)

                else:
                    # Handle other types of paths (e.g., symlinks)
                    msg = f"Deleted '{matched_path}' (unsupported path type)."
                    HandleError(msg, caller_filename, caller_line)
            except FileNotFoundError:
                if verbose:
                    msg=(f"File or folder not found: '{matched_path}'. Skipping deletion.")
                    ShowWarning(msg, caller_filename, caller_line)
            except Exception as e:
                if verbose:
                    msg=(f"Error while deleting '{matched_path}': {str(e)}")
                    ShowWarning(msg, caller_filename, caller_line)

    except Exception as e:
        HandleError(str(e), caller_filename, caller_line)
        
def Rename(src_path=None, new_name=None, verbose=True):
    """
    Renames a file or folder based on the given source path and new name.

    Args:
        src_path (str): The source path of the file or folder to be renamed.
        new_name (str): The new name for the file or folder.
        verbose (bool): Whether to display verbose messages. Defaults to True.
    """
    check_required_args()
    caller_filename, caller_line = get_caller_info()

    try:
        if not os.path.isabs(src_path):
            # If src_path is a relative path, make it relative to the current working directory
            src_path = os.path.join(os.getcwd(), src_path)

        if os.path.exists(src_path):
            parent_folder = os.path.dirname(src_path)
            new_path = os.path.join(parent_folder, new_name)
            os.rename(src_path, new_path)
            if verbose:
                msg=(f"Renamed '{src_path}' to '{new_path}'.")
                ShowInfo(msg, caller_filename, caller_line)

        else:
            msg = f"No output ('{src_path}' does not exist)."
            HandleError(msg, caller_filename, caller_line)
    except Exception as e:
        msg = f"[Error] {str(e)}"
        HandleError(msg, caller_filename, caller_line)


def CreateFolder(path=None, mode="a", verbose=True):
    """
    Create a folder with the given path using one of the following modes:

    - 'f' (force_create): Deletes the folder if it already exists and then creates it.
    - 'o' (overwrite): Overwrites the folder if it already exists.
    - 'c' (create_if_not_exist): Creates the folder if it doesn't exist.
    - 'a' (ask_user): Asks the user whether to delete and overwrite, or just pass if the folder exists.

    Args:
        path (str): The path to the folder to be created.
        mode (str): The mode for folder creation ('f', 'o', 'c', or 'a'). Defaults to 'a' (ask_user).
        verbose (bool): Whether to display verbose messages. Defaults to True.
    """
    check_required_args()
    caller_filename, caller_line=get_caller_info()
        
    try:

        
        if mode == "f":
            if os.path.exists(path):
                if verbose:
                    msg=(f"The folder '{path}' already exists. Deleting and recreating.")
                    ShowInfo(msg, caller_filename, caller_line)
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
        elif mode == "o":
            if os.path.exists(path):
                if verbose:
                    msg=(f"The folder '{path}' already exists. Overwriting.")
                    ShowInfo(msg, caller_filename, caller_line)
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
        elif mode == "c":
            if not os.path.exists(path):
                if verbose:
                    msg=(f"The folder '{path}' doesn't exist. Creating it.")
                    ShowInfo(msg, caller_filename, caller_line)
                os.makedirs(path)
            else:
                if verbose:
                    msg=(f"The folder '{path}' already exists. Skipping creation.")
                    ShowInfo(msg, caller_filename, caller_line)
                    
        elif mode == "a":
            if os.path.exists(path):
                user_input = input(f"❓❓ The folder '{path}' already exists. Do you want to delete and overwrite it? (y/n): ")
                if user_input.lower() == "y":
                    if verbose:
                        msg=(f"The folder '{path}' already exists. Deleting and recreating.")
                        ShowInfo(msg, caller_filename, caller_line)
                    shutil.rmtree(path)
                    os.makedirs(path, exist_ok=True)
                else:
                    msg=("Folder creation aborted.")
                    ShowInfo(msg, caller_filename, caller_line)
            else:
                os.makedirs(path, exist_ok=True)
        else:
            msg=f"Invalid mode '{mode}'. Please use 'f' (force_create), 'o' (overwrite), 'c' (create_if_not_exist), or 'a' (ask_user)."
            HandleError(msg,caller_filename, caller_line)

    except PermissionError as pe:
        msg=f"Error: Permission denied to create the folder '{path}' (Occurred in {caller_filename}, line {caller_line})"
        HandleError(msg,caller_filename, caller_line)
    except Exception as e:
        msg=f"Error: {str(e)} (Occurred in {caller_filename}, line {caller_line})"
        HandleError(msg,caller_filename, caller_line)
      

file_pointers = {}

def save_file_pointer(file_path, offset):
    file_pointers[file_path] = offset

def get_file_pointer(file_path):
    return file_pointers.get(file_path, 0)

def reset_file_pointer(file_path):
    file_pointers[file_path] = 0  # Explicitly set to 0
    # or
    # file_pointers.pop(file_path, None)  # Remove the file path entry

def ReadFile(file_path):
    check_required_args()
    caller_filename, caller_line=get_caller_info()
         
    try:
        if not os.path.exists(file_path):
            msg=f"File not found: {file_path}"
            HandleError(msg,caller_filename, caller_line)

        offset = get_file_pointer(file_path)
        with open(file_path, 'r') as file:
            file.seek(offset)
            line = file.readline()
            if not line:                        
                reset_file_pointer(file_path)  # Reset the pointer on completion              
                return None

            # Strip various newline characters (\n, \r, \r\n)
            line = line.rstrip('\n').rstrip('\r')
            save_file_pointer(file_path, file.tell())
            return line

    except PermissionError as pe:
        msg=f"Error: Permission denied to read the file '{file_path}' (Occurred in {caller_filename}, line {caller_line})"
        HandleError(msg,caller_filename, caller_line)
    except FileNotFoundError as fe:
        msg=f"Error: {str(fe)} (Occurred in {caller_filename}, line {caller_line})"
        HandleError(msg,caller_filename, caller_line)
    except Exception as e:
        msg=f"Error: {str(e)} (Occurred in {caller_filename}, line {caller_line})"
        HandleError(msg,caller_filename, caller_line)


def save_file_pointer(file_path, offset):
    file_pointers[file_path] = offset

def get_file_pointer(file_path):
    return file_pointers.get(file_path, 0)

def WriteFile(file_path=None, lines=None):
    """
    Write lines to a file in either append or write mode based on the file pointer.

    Args:
        file_path (str): The path to the file to be written.
        lines (str or list of str): The lines to be written to the file.

    Returns:
        None
    """
    check_required_args()
    caller_filename, caller_line=get_caller_info()
          
    try:

        offset = get_file_pointer(file_path)

        if offset == 0:
            mode = "w"  # If the file pointer is at the beginning, use write mode
        else:
            mode = "a+"  # If the file pointer is not at the beginning, use append and read mode

        with open(file_path, mode) as file:
            # Move the file pointer to the end of the file
            file.seek(0, os.SEEK_END)

            if isinstance(lines, str):
                file.write(lines)
            elif isinstance(lines, list):
                file.writelines(lines)

            # Update the file pointer to the end of the file
            save_file_pointer(file_path, file.tell())

    except Exception as e:
        msg=f"{e}"
        HandleError(msg,caller_filename, caller_line)




def ReadImage(image_path=None, mode='RGB', method='auto'):

    """
    Read an image from the specified file path.

    Args:
        image_path (str): The path to the image file.
        mode (str): The desired mode for loading the image ('RGB', 'L', etc.). Defaults to 'RGB'.
        method (str): The method to use for loading the image ('auto', 'PIL', or 'CV2'). Defaults to 'auto'.

    Returns:
        PIL.Image.Image or numpy.ndarray: The loaded image.
    """
    check_required_args()
    caller_filename, caller_line=get_caller_info()
        
    try:
        # Check if mode is valid
        if mode not in ['RGB', 'L']:
            msg = "Invalid mode. Please use 'RGB' or 'L'."
            HandleError(msg, caller_filename, caller_line)
                
        # Determine the appropriate method for image loading (PIL or CV2) based on file extension
        if method == 'auto':
            _, file_extension = os.path.splitext(image_path)
            if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                method = 'PIL'
            else:
                method = 'CV2'

        # Load image using PIL (Pillow)
        if method == 'PIL':
            img = Image.open(image_path)
            # Convert grayscale image to RGB if specified
            if img.mode == 'L' and mode == 'RGB':
                img = img.convert('RGB')
        
        # Load image using OpenCV (cv2)
        elif method == 'CV2':
            img = cv2.imread(image_path)
            if img is None:
                msg=f"File not found or unsupported format: {image_path}"
                HandleError(msg,caller_filename, caller_line)

            # Handle grayscale and color conversions using OpenCV
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                if mode == 'RGB':
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Raise error for unsupported methods
        else:
            msg=f"Unsupported method: {method}. Please use 'auto', 'PIL', or 'CV2'."
            HandleError(msg,caller_filename, caller_line)

        return img

    except PermissionError as pe:
        msg=f"Error: Permission denied to open - {image_path}"
        HandleError(msg,caller_filename, caller_line)
        exit(0)

    except FileNotFoundError as e:
        msg=f"Error: {str(e)}"
        HandleError(msg,caller_filename, caller_line)
        exit(0)

    except Exception as e:        
        msg=f"Error reading the image: {str(e)})"
        HandleError(msg,caller_filename, caller_line)
        exit(0)
    return None

def SaveImage(image=None, save_path=None, method='auto'):
    save_path = os.path.abspath(save_path)

    check_required_args()
    caller_filename, caller_line = get_caller_info()
        
    try:
        if method == 'auto':
            if isinstance(image, Image.Image):
                method = 'PIL'
            elif isinstance(image, np.ndarray):
                method = 'CV2'
            else:
                msg = "Unsupported image type for automatic saving method detection."
                HandleError(msg, caller_filename, caller_line)

        if method == 'PIL':
            if isinstance(image, Image.Image):
                image.save(save_path)
            else:
                msg = "Unsupported image type. Please provide a PIL Image."
                HandleError(msg, caller_filename, caller_line)

        elif method == 'CV2':
            if isinstance(image, np.ndarray):
                # Ensure the correct number of color channels (e.g., convert grayscale to RGB if needed)
                if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                    if image.shape[2] == 1:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif image.shape[2] == 4:
                        # Handle RGBA images by converting them to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif len(image.shape) == 3 and image.shape[2] != 3:
                    msg = "Unsupported number of channels in input image."
                    HandleError(msg, caller_filename, caller_line)
                
                # Check if the user has write permission to the save_path
                if not os.access(os.path.dirname(save_path), os.W_OK):
                    msg = f"Permission denied to save the image to {save_path}"
                    HandleError(msg, caller_filename, caller_line)
                cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                msg = "Unsupported image type. Please provide a numpy array (cv2 image)."
                HandleError(msg, caller_filename, caller_line)

        else:
            msg = f"Unsupported method: {method}. Please use 'auto', 'PIL', or 'CV2'."
            HandleError(msg, caller_filename, caller_line)

        return True  # Image saved successfully
    except PermissionError as pe:
        msg = f"PermissionError: {str(pe)}"
        HandleError(msg, caller_filename, caller_line)
        return False
    except ValueError as ve:
        msg = f"ValueError: {str(ve)}"
        HandleError(msg, caller_filename, caller_line)
        return False
    except Exception as e:
        msg = f"Error saving the image: {str(e)}"
        HandleError(msg, caller_filename, caller_line)
        return False



def ConvertToGrayscale(image=None, method='auto'):
    """
    Convert an image to grayscale.

    Args:
        image (PIL.Image.Image or numpy.ndarray): The input image.
        method (str): The method to use for converting the image ('auto', 'PIL', or 'CV2').
                      Defaults to 'auto' which automatically detects the input type.

    Returns:
        PIL.Image.Image or numpy.ndarray: The grayscale image.

    Raises:
        ValueError: If an unsupported method is specified.

    Example:
        # Convert an image to grayscale using the default 'auto' method
        grayscale_image = ConvertToGrayscale(image)

        # Convert an image to grayscale using the 'PIL' method
        grayscale_image = ConvertToGrayscale(image, method='PIL')

        # Convert an image to grayscale using the 'CV2' method
        grayscale_image = ConvertToGrayscale(image, method='CV2')
    """
    check_required_args()
    caller_filename, caller_line=get_caller_info()
          
    try:
   
        if method == 'auto':
            if isinstance(image, Image.Image):
                return image.convert('L')
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    return image
                elif len(image.shape) == 3 and image.shape[2] == 3:
                    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    msg="Unsupported image format for automatic conversion to grayscale."
                    HandleError(msg,caller_filename, caller_line)
            else:
                msg="Unsupported image type. Please provide a PIL Image or numpy array (cv2 image)."
                HandleError(msg,caller_filename, caller_line)
        elif method == 'PIL':
            if isinstance(image, Image.Image):
                return image.convert('L')
            else:
                msg="Unsupported image type for 'PIL' method. Please provide a PIL Image."
                HandleError(msg,caller_filename, caller_line)
        elif method == 'CV2':
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    return image
                elif len(image.shape) == 3 and image.shape[2] == 3:
                    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    msg="Unsupported image format for 'CV2' conversion to grayscale."
                    HandleError(msg,caller_filename, caller_line)
            else:
                msg="Unsupported image type for 'CV2' method. Please provide a numpy array (cv2 image)."
                HandleError(msg,caller_filename, caller_line)
        else:
            msg=f"Unsupported method: {method}. Please use 'auto', 'PIL', or 'CV2'."
            HandleError(msg,caller_filename, caller_line)
    except Exception as e:
        msg=f"Error converting to grayscale: {str(e)}"
        HandleError(msg,caller_filename, caller_line)
        return None

def ConvertToRGB(image=None, method='auto'):
    """
    Convert an image to RGB color mode.

    Args:
        image (PIL.Image.Image or numpy.ndarray): The input image.
        method (str): The method to use for conversion ('auto', 'PIL', or 'CV2').
                      Defaults to 'auto' which automatically detects the input type.

    Returns:
        PIL.Image.Image or numpy.ndarray: The image converted to RGB color mode.

    Raises:
        ValueError: If an unsupported method is specified.

    Example:
        # Convert an image to RGB color mode using the default 'auto' method
        rgb_image = ConvertToRGB(image)

        # Convert an image to RGB color mode using the 'PIL' method
        rgb_image = ConvertToRGB(image, method='PIL')

        # Convert an image to RGB color mode using the 'CV2' method
        rgb_image = ConvertToRGB(image, method='CV2')
    """
    check_required_args()
    caller_filename, caller_line=get_caller_info()
              
    try:
        if method == 'auto':
            if isinstance(image, Image.Image):
                if image.mode == 'L':
                    return image.convert('RGB')
                elif image.mode == 'RGB':
                    return image
                else:
                    msg="Unsupported image mode for automatic conversion to RGB."
                    HandleError(msg,caller_filename, caller_line)
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    msg="Cannot convert a grayscale image with 'auto' method."
                    HandleError(msg,caller_filename, caller_line)
                elif len(image.shape) == 3 and image.shape[2] == 3:
                    return image
                elif len(image.shape) == 3 and image.shape[2] == 1:
                    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    msg="Unsupported image format for automatic conversion to RGB."
                    HandleError(msg,caller_filename, caller_line)
            else:
                msg="Unsupported image type. Please provide a PIL Image or numpy array (cv2 image)."
                HandleError(msg,caller_filename, caller_line)
        elif method == 'PIL':
            if isinstance(image, Image.Image):
                if image.mode == 'L':
                    return image.convert('RGB')
                elif image.mode == 'RGB':
                    return image
                else:
                    msg="Unsupported image mode for 'PIL' conversion to RGB."
                    HandleError(msg,caller_filename, caller_line)
            else:
                msg="Unsupported image type for 'PIL' method. Please provide a PIL Image."
                HandleError(msg,caller_filename, caller_line)
        elif method == 'CV2':
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    msg="Cannot convert a grayscale image with 'CV2' method."
                    HandleError(msg,caller_filename, caller_line)
                elif len(image.shape) == 3 and image.shape[2] == 3:
                    return image
                elif len(image.shape) == 3 and image.shape[2] == 1:
                    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    msg="Unsupported image format for 'CV2' conversion to RGB."
                    HandleError(msg,caller_filename, caller_line)
            else:
                msg="Unsupported image type for 'CV2' method. Please provide a numpy array (cv2 image)."
                HandleError(msg,caller_filename, caller_line)
        else:
            msg=f"Unsupported method: {method}. Please use 'auto', 'PIL', or 'CV2'."
            HandleError(msg,caller_filename, caller_line)
    except Exception as e:
        msg=f"Error converting the image to RGB: {str(e)}"
        HandleError(msg,caller_filename, caller_line)
        return None

def CropImage(image=None, coordinates=None):
    """
    Crop an image.

    Args:
        image (PIL.Image.Image): The input image.
        coordinates (list): A list containing the left, top, right, and bottom coordinates.

    Returns:
        PIL.Image.Image: The cropped image.

    Example:
        # Crop a region of interest from an image
        cropped_image = CropImage(image, [0, 0, 50, 50])
    """
    check_required_args()
    caller_filename, caller_line=get_caller_info()
          
    try:
        if isinstance(image, Image.Image):
            if len(coordinates) == 4:
                left, top, right, bottom = coordinates
                return image.crop((left, top, right, bottom))
            else:
                msg="Coordinates list should contain exactly 4 values."
                HandleError(msg,caller_filename, caller_line)
        else:
            msg="Unsupported image type. Please provide a PIL Image."
            HandleError(msg,caller_filename, caller_line)
    except Exception as e:        
        msg=f"Error cropping the image: {str(e)}"
        HandleError(msg,caller_filename, caller_line)
        return None


def GetImageSize(image=None, method='auto'):
    """
    Get the size (width, height) and number of channels of an image using either PIL (Pillow) or OpenCV (cv2).

    Args:
        image (PIL.Image.Image or numpy.ndarray): The image to get the size and channels from.
        method (str): The method to use for reading the image ('auto', 'PIL' for Pillow, 'CV2' for OpenCV).
                      Defaults to 'auto'.

    Returns:
        tuple: A tuple containing the width, height, and number of channels of the image, e.g., (width, height, channels).
               If the image method is unsupported, returns (0, 0, 0).
    """
    check_required_args()
    caller_filename, caller_line=get_caller_info()
        
    try:
        if method == 'auto':
            if isinstance(image, Image.Image):
                # Get the size (width and height) of the image
                width, height = image.size
                # Get the number of channels (always 3 for RGB images)
                channels = 3
            elif isinstance(image, np.ndarray):
                # Get the size (width and height) of the image
                height, width, channels = image.shape
            else:
                msg="Unsupported image type for automatic detection."
                HandleError(msg,caller_filename, caller_line)
        elif method == 'PIL':
            if isinstance(image, Image.Image):
                # Get the size (width and height) of the image
                width, height = image.size
                # Get the number of channels (always 3 for RGB images)
                channels = 3
            else:
                msg="Unsupported image type for 'PIL' method. Please provide a PIL Image."
                HandleError(msg,caller_filename, caller_line)
        elif method == 'CV2':
            if isinstance(image, np.ndarray):
                # Get the size (width and height) of the image
                height, width, channels = image.shape
            else:
                msg="Unsupported image type for 'CV2' method. Please provide a numpy array (cv2 image)."
                HandleError(msg,caller_filename, caller_line)
        else:
            msg=f"Unsupported method: {method}. Please use 'auto', 'PIL', or 'CV2'."
            HandleError(msg,caller_filename, caller_line)

        return width, height, channels

    except Exception as e:
        msg=f"Error: {str(e)}"
        HandleError(msg,caller_filename, caller_line)
        return 0, 0, 0

def ResizeImage(image=None, size=None, verbose=True, interpolation='IANTIALIAS'):
    """
    Resize an image (PIL or cv2) to the specified size while preserving the aspect ratio.

    Args:
        image (PIL.Image.Image or numpy.ndarray): The input image (PIL or cv2 format).
        size (tuple): The target size (width, height).
        verbose (bool): Whether to display verbose messages. Defaults to True.
        interpolation: The interpolation method to use (shorthand or full name).
            - For PIL images, options are: NB, IBOX, IBILINEAR, IHAMMING, IBICUBIC, ILANCZOS, IANTIALIAS.
            - For cv2 images, options are: CV_NEAREST, CV_LINEAR, CV_CUBIC, CV_LANCZOS4.
            
    Returns:
        PIL.Image.Image or numpy.ndarray: The resized image (PIL or cv2 format).
    """
    check_required_args()
    caller_filename, caller_line = get_caller_info()
    
    
    # Define a dictionary to map shorthand names to full interpolation names
    INTERPOLATION_MAP = {
        'NB': Image.NEAREST,
        'IBOX': Image.BOX,
        'IBILINEAR': Image.BILINEAR,
        'IHAMMING': Image.HAMMING,
        'IBICUBIC': Image.BICUBIC,
        'ILANCZOS': Image.LANCZOS,
        'IANTIALIAS': Image.ANTIALIAS,
        'CV_NEAREST': cv2.INTER_NEAREST,
        'CV_LINEAR': cv2.INTER_LINEAR,
        'CV_CUBIC': cv2.INTER_CUBIC,
        'CV_LANCZOS4': cv2.INTER_LANCZOS4,
    }

              
    try:
        if isinstance(image, Image.Image):  # PIL image
            if not isinstance(size, tuple) or len(size) != 2:
                msg = "Input 'size' must be a tuple of two integers (width, height)."
                HandleError(msg, caller_filename, caller_line)

            if verbose:
                print(f"Resizing PIL image to {size} using interpolation method: {interpolation}...")
            
            if interpolation in INTERPOLATION_MAP:
                interpolation = INTERPOLATION_MAP[interpolation]

            resized_image = image.resize(size, interpolation)

        elif isinstance(image, np.ndarray):  # cv2 image
            if verbose:
                print(f"Resizing cv2 image to {size} using interpolation method: {interpolation}...")

            if interpolation in INTERPOLATION_MAP:
                interpolation = INTERPOLATION_MAP[interpolation]

            if interpolation not in [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]:
                msg = "Invalid interpolation method for cv2 image. Using cv2.INTER_LINEAR by default."
                HandleError(msg, caller_filename, caller_line)
                interpolation = cv2.INTER_LINEAR

            resized_image = cv2.resize(image, size, interpolation=interpolation)

        else:
            msg = "Input 'image' must be a PIL Image object or a numpy.ndarray (cv2 image)."
            HandleError(msg, caller_filename, caller_line)
        
        return resized_image

    except Exception as e:
        msg = f"{e}"
        HandleError(msg, caller_filename, caller_line)
        exit(1)


def GaussianBlurImage(image=None, sigma=1.0, verbose=True):
    """
    Apply Gaussian blur to an image.

    Args:
        image (PIL.Image.Image): The input image.
        sigma (float): The standard deviation of the Gaussian kernel.
        verbose (bool): Whether to display verbose messages. Defaults to True.

    Returns:
        PIL.Image.Image: The blurred image.
    """
    
    check_required_args()
    caller_filename, caller_line=get_caller_info()
          
    try:
        if not isinstance(image, Image.Image):
            msg="Input 'image' must be a PIL Image object."
        
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            msg="Input 'sigma' must be a positive number."
            HandleError(msg,caller_filename, caller_line)
        
        if verbose:
            msg=(f"Applying Gaussian blur with sigma={sigma}...")
            ShowInfo(msg,caller_filename, caller_line)

        
        blurred_image = image.filter(ImageFilter.GaussianBlur(sigma))
        return blurred_image
    except Exception as e:
        msg=f"{e}"
        HandleError(msg,caller_filename, caller_line)
        
        exit(1)


def ConvertImageToGrayscale(image=None, verbose=True):
    """
    Convert an image to grayscale.

    Args:
        image (PIL.Image.Image): The input image.
        verbose (bool): Whether to display verbose messages. Defaults to True.

    Returns:
        PIL.Image.Image: The grayscale image.
    """
    
    check_required_args()
    caller_filename, caller_line=get_caller_info()
               
    try:
        if not isinstance(image, Image.Image):
            msg="Input 'image' must be a PIL Image object."
            HandleError(msg,caller_filename, caller_line)
        
        if verbose:
            print("Converting image to grayscale...")
        
        grayscale_image = image.convert("L")
        return grayscale_image
    except Exception as e:
        msg=f"{e}"
        HandleError(msg,caller_filename, caller_line)        
        exit(1)


def SharpenImage(image=None, factor=2.0, verbose=True):
    """
    Sharpen an image.

    Args:
        image (PIL.Image.Image): The input image.
        factor (float): The sharpening factor.
        verbose (bool): Whether to display verbose messages. Defaults to True.

    Returns:
        PIL.Image.Image: The sharpened image.
    """
    
    check_required_args()
    caller_filename, caller_line=get_caller_info()
             
    try:
        if not isinstance(image, Image.Image):
            msg="Input 'image' must be a PIL Image object."
            HandleError(msg,caller_filename, caller_line)
        
        if not isinstance(factor, (int, float)) or factor <= 0:
            msg="Input 'factor' must be a positive number."
            HandleError(msg,caller_filename, caller_line)
        
        if verbose:
            msg=(f"Sharpening image with factor={factor}...")
            ShowInfo(msg,caller_filename, caller_line)
        
        enhancer = ImageEnhance.Sharpness(image)
        sharpened_image = enhancer.enhance(factor)
        return sharpened_image
    except Exception as e:
        msg=f"{e}"
        HandleError(msg,caller_filename, caller_line)          
        

def DetectEdgesInImage(image=None, method='canny', threshold1=100, threshold2=200, verbose=True):
    """
    Detect edges in an image using various edge detection methods.

    Args:
        image (PIL.Image.Image): The input image.
        method (str): The edge detection method to use ('canny', 'sobel', 'laplacian', 'prewitt', or 'scharr').
        threshold1 (int): The first threshold for the hysteresis procedure (only for 'canny' method).
        threshold2 (int): The second threshold for the hysteresis procedure (only for 'canny' method).
        verbose (bool): Whether to display verbose messages. Defaults to True.

    Returns:
        PIL.Image.Image: The edge-detected image.
    """
    check_required_args()
    caller_filename, caller_line=get_caller_info()
        
    try:
        if not isinstance(image, Image.Image):
            msg = "Input 'image' must be a PIL Image object."
            HandleError(msg, caller_filename, caller_line)

        if method == 'canny':
            if not isinstance(threshold1, int) or not isinstance(threshold2, int) or threshold1 >= threshold2:
                msg = "For 'canny' method, input thresholds must be integers, and threshold1 must be less than threshold2."
                HandleError(msg, caller_filename, caller_line)

            if verbose:
                print(f"Detecting edges in image using Canny edge detection (threshold1={threshold1}, threshold2={threshold2})...")

            # Convert the input image to grayscale
            image = image.convert('L')
            image_array = np.array(image)
            edges = cv2.Canny(image_array, threshold1, threshold2)
            edge_image = Image.fromarray(edges)
            return edge_image

        elif method == 'sobel' or method == 'laplacian' or method == 'prewitt' or method == 'scharr':
            if verbose:
                print(f"Detecting edges in image using {method.capitalize()} edge detection...")
            
            # Convert the input image to a NumPy array
            image_array = np.array(image)

            # Convert the image to grayscale if it's not already
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

            if method == 'sobel':
                edges = cv2.Sobel(image_array, cv2.CV_64F, 1, 1)
            elif method == 'laplacian':
                edges = cv2.Laplacian(image_array, cv2.CV_64F)
            elif method == 'prewitt':
                kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
                gradient_x = cv2.filter2D(image_array, -1, kernel_x)
                gradient_y = cv2.filter2D(image_array, -1, kernel_y)
                edges = np.sqrt(gradient_x**2 + gradient_y**2)
            elif method == 'scharr':
                gradient_x = cv2.Scharr(image_array, cv2.CV_64F, 1, 0)
                gradient_y = cv2.Scharr(image_array, cv2.CV_64F, 0, 1)
                edges = np.sqrt(gradient_x**2 + gradient_y**2)

            edge_image = Image.fromarray(edges.astype('uint8'))
            return edge_image

        else:
            msg = f"Unsupported edge detection method: {method}. Please use 'canny', 'sobel', 'laplacian', 'prewitt', or 'scharr'."
            HandleError(msg, caller_filename, caller_line)

    except Exception as e:
        msg = f"{e}"
        HandleError(msg, caller_filename, caller_line)

def ConvolveImage(image=None, kernel=None, verbose=True):
    """
    Apply convolution to an image with a given kernel.

    Args:
        image (PIL.Image.Image): The input image.
        kernel (numpy.ndarray): The convolution kernel.
        verbose (bool): Whether to display verbose messages. Defaults to True.

    Returns:
        PIL.Image.Image: The convolved image.
    """
    check_required_args()
    caller_filename, caller_line=get_caller_info()
           
    try:
        if not isinstance(image, Image.Image):
            msg="Input 'image' must be a PIL Image object."
            HandleError(msg,caller_filename, caller_line)
        
        if not isinstance(kernel, np.ndarray) or len(kernel.shape) != 2:
            msg="Input 'kernel' must be a 2D NumPy array."
            HandleError(msg,caller_filename, caller_line)
        
        if verbose:
            print("Applying convolution to image...")
        
        image_array = np.array(image)
        convolved_image = convolve2d(image_array, kernel, mode='same', boundary='wrap')
        convolved_image = Image.fromarray(convolved_image)
        return convolved_image
    except Exception as e:
        msg=f"{e}"
        HandleError(msg,caller_filename, caller_line)          
        exit(1)



def ApplyFilter(image=None, kernel=None):
    """
    Apply a convolution filter to an image using a custom kernel.

    Args:
        image (PIL.Image.Image or numpy.ndarray): The input image to apply the filter to.
        kernel (numpy.ndarray): The custom convolution kernel.

    Returns:
        PIL.Image.Image or numpy.ndarray: The filtered image.
    """
    check_required_args()
    caller_filename, caller_line=get_caller_info()
             
    try:
        if isinstance(image, Image.Image):
            # If the input image is a PIL Image, convert it to a numpy array
            image = np.array(image)

        if isinstance(image, np.ndarray):
            # If the input image is a numpy array (cv2 image)
            filtered_image = cv2.filter2D(image, -1, kernel)

            if len(filtered_image.shape) == 2:
                # Convert grayscale image back to PIL Image
                filtered_image = Image.fromarray(filtered_image)
                return filtered_image

            return filtered_image

        msg="Unsupported image type. Please provide a PIL Image or numpy array (cv2 image)."
        HandleError(msg,caller_filename, caller_line)

    except Exception as e:
        msg=f"Error applying the filter: {str(e)}"
        HandleError(msg,caller_filename, caller_line)



def ShowImage(image=None, title="Image", verbose=True):
    """
    Display an image using matplotlib.

    Args:
        image (PIL.Image.Image): The input image.
        title (str): The title of the displayed image.
        verbose (bool): Whether to display verbose messages. Defaults to True.

    Returns:
        None
    """
    check_required_args()
    caller_filename, caller_line=get_caller_info()
           
    try:
        if not isinstance(image, Image.Image):
            msg="Input 'image' must be a PIL Image object."
        
        if verbose:
            print("Displaying image...")
        
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
    except Exception as e:
        msg=f"{e}"
        HandleError(msg,caller_filename, caller_line)          
        
        exit(1)

def CV2PIL(cv2_image=None):
    """
    Convert an OpenCV image (BGR format) to a PIL Image (RGB format).

    Args:
        cv2_image (numpy.ndarray): The OpenCV image.

    Returns:
        PIL.Image.Image or None: The PIL Image if conversion is successful, None otherwise.
    """
    check_required_args()
    caller_filename, caller_line=get_caller_info()
        
    try:
        if cv2_image is None:
            return None
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    except Exception as e:
        msg = f"Error converting from OpenCV to PIL: {str(e)}"
        HandleError(msg, caller_filename, caller_line)
        return None

def PIL2CV2(pil_image=None):
    """
    Convert a PIL Image (RGB format) to an OpenCV image (BGR format).

    Args:
        pil_image (PIL.Image.Image): The PIL Image.

    Returns:
        numpy.ndarray or None: The OpenCV image if conversion is successful, None otherwise.
    """
    
    check_required_args()

    caller_frame = sys._getframe(1)
    caller_line = caller_frame.f_lineno
    caller_filename = caller_frame.f_globals.get('__file__')

    try:
        if pil_image is None:
            return None
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        msg = f"Error converting from PIL to OpenCV: {str(e)}"
        HandleError(msg, caller_filename, caller_line)
        return None
    
    
def copy_brighter_pixels(np_img1, np_img2):
    """
    Takes two numpy arrays representing images, compares their pixel brightness, 
    and copies the brighter pixels from image 2 to image 1.
    
    :param np_img1: Numpy array of the first image (destination)
    :param np_img2: Numpy array of the second image (source)
    :return: Numpy array of the modified first image
    """
    # Calculate the brightness of each pixel using the luminosity method
    brightness_img1 = np_img1[..., 0]*0.2989 + np_img1[..., 1]*0.5870 + np_img1[..., 2]*0.1140
    brightness_img2 = np_img2[..., 0]*0.2989 + np_img2[..., 1]*0.5870 + np_img2[..., 2]*0.1140

    # Create a mask where the brightness of img2 is greater than img1
    mask = brightness_img2 > brightness_img1

    # Copy brighter pixels from np_img2 to np_img1
    np_img1[mask] = np_img2[mask]

    return np_img1

def copy_brighter_pixels_percentage(np_img1, np_img2, percentage=50):
    # Check if images have three dimensions (height, width, channels)
    if np_img1.ndim != 3 or np_img2.ndim != 3:
        raise ValueError("Both images must have three dimensions [height, width, channels]")
    
    # Check if both images have the same shape
    if np_img1.shape != np_img2.shape:
        raise ValueError("Both images must have the same shape")
    
    # Calculate the brightness of each pixel using the luminosity method
    brightness_img1 = np.sum(np_img1 * [0.2989, 0.5870, 0.1140], axis=2)
    brightness_img2 = np.sum(np_img2 * [0.2989, 0.5870, 0.1140], axis=2)

    # Create a mask where the brightness of img2 is greater than img1
    mask = brightness_img2 > brightness_img1

    # Get the indices of the mask where the condition is True
    brighter_indices = np.argwhere(mask)

    # Determine the number of pixels to copy over
    num_pixels_to_copy = int(len(brighter_indices) * (percentage / 100.0))

    # Select a random subset of these indices
    selected_indices = np.random.choice(len(brighter_indices), size=num_pixels_to_copy, replace=False)

    # Copy the selected brighter pixels from np_img2 to np_img1
    for idx in selected_indices:
        np_img1[tuple(brighter_indices[idx])] = np_img2[tuple(brighter_indices[idx])]

    return np_img1


def create_brighter_image(np_img1, np_img2):
    """
    Takes two numpy arrays representing images, compares them pixel by pixel across
    all channels, and creates a new image array with the brighter pixels from each image.
    
    :param np_img1: Numpy array of the first image
    :param np_img2: Numpy array of the second image
    :return: Numpy array of the resultant image
    """
    # Calculate the brightness of each pixel using the luminosity method
    brightness_img1 = np_img1[..., 0]*0.2989 + np_img1[..., 1]*0.5870 + np_img1[..., 2]*0.1140
    brightness_img2 = np_img2[..., 0]*0.2989 + np_img2[..., 1]*0.5870 + np_img2[..., 2]*0.1140

    # Create a mask where the brightness of img1 is greater than img2
    mask = brightness_img1 > brightness_img2

    # Initialize an empty image array with the same shape and type as np_img1/np_img2
    result_image = np.zeros_like(np_img1)

    # Use the mask to select pixels from np_img1 and np_img2
    result_image[mask] = np_img1[mask]    # If img1 is brighter, take from img1
    result_image[~mask] = np_img2[~mask]  # Else, take from img2

    # Convert the result to uint8 if needed
    result_image_uint8 = result_image.astype(np.uint8)

    return result_image_uint8
            
    
