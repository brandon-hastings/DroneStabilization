import os
import shutil
from pathlib import Path

'''make subdirectories'''


def make_folder(project_folder, folder_name):
    new_folder = folder_name
    if new_folder in os.listdir(project_folder):
        shutil.rmtree(new_folder)
    new_dir = os.path.join(project_folder, new_folder)
    os.makedirs(new_dir)
    return new_dir


'''find image directories that still need processed in selected step. Runs at beginning of each major function call.
Inputs:
config - project config file
new_folder_name - basename of folder where modified images will be saved
search_folder_name - basename of folder storing images to be modified by outer function
Outputs:
folders_to_process - list of folders that are not in save folder location, thus they need processed
save_folder - folder path as Posix Path where image directories are saved to'''


def search_existing_directories(config, new_folder_name, search_folder_name):
    image_directories = config["image_folders"]
    working_folder = Path(config["project_path"]) / search_folder_name
    save_folder = Path(config["project_path"]) / new_folder_name

    if os.path.exists(working_folder):
        folders_to_process = []
        if os.path.exists(save_folder):
            processed_folders = [os.path.basename(i) for i in os.listdir(save_folder) if os.path.isdir(i)]
            for i in image_directories:
                if os.path.basename(i) not in processed_folders:
                    folders_to_process.append(working_folder / os.path.basename(i))
        else:
            os.makedirs(save_folder)
            for i in image_directories:
                folders_to_process.append(working_folder / os.path.basename(i))
        return folders_to_process, save_folder
    else:
        # TODO: Make more case specific error messages, as some steps (background segmentation) are optional
        print("Previous step not completed")
        raise FileNotFoundError(
            "Directory {} was not found, perhaps the previous step has not been completed?".format(working_folder)
        )


'''given a string file path, find a given folder path and replace it with another folder path
Inputs:
file_path - Path object, str - file path object or string. Path will be normalized in function
find - str - directory to find in given file_path
replace - str - directory to replace in given file path
Outputs:
new_path - os path object'''


def replace_path(file_path, find, replace):
    # normalize given path
    path = os.path.normpath(str(file_path))
    # split path on OS specific separator
    split_path = path.split(os.sep)
    # first separator gets removed, adding back
    split_path.insert(0, os.sep)
    # replace file path section via insert then remove
    split_path.insert(split_path.index(find), replace)
    split_path.remove(find)
    # join and normalize new path
    new_path = os.path.normpath(os.path.join(*split_path))
    return new_path


'''find subdirectories in given folder, ignoring files, and output to list.
Trivial function, I just got tired of typing a list generator when needed
Input:
folder - folder to find directories in
Output:
list of directories in folder'''


def find_directories(folder):
    return [i for i in os.scandir(folder) if os.path.isdir(i)]


'''remove trailing slash in case of user input, not a problem with file selection'''


def correct_path(path):
    return os.path.normpath(path)
