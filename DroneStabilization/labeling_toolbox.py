import cv2
import rawpy
import pickle as pkl
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter.messagebox import askyesno
from tkinter.simpledialog import askstring
from tkinter.filedialog import askdirectory
from PIL import Image, ImageTk
from pathlib import Path
import matplotlib as mpl
from glob import glob
import random

# save dictionary created after each folder is analyzed to a csv
from utils import directory_utils, yaml_utils

'''If user indicates they will be using machine learning workflows, select images to label'''


def select_images_ml(config, image_type, search_folder, output_folder) -> list:
    f"""glob to find all images in necessary folder. Because file extensions vary in case between OS and glob is case
    sensitive on Linux, construct a string of upper and lower characters in image_type to test. Returns all images of
    given image_type in search_folder"""

    # define pickle file name to save images list
    pkl_file = os.path.join(output_folder, "ml_selected_images.pkl")

    # ensure image_type is string
    image_type = str(image_type)
    # list to hold upper and lower characters of file type
    glob_file_suffix = []
    # handle if passed image_type has a leading period in string, change start range for next loop
    if image_type[0] == ".":
        glob_file_suffix.append(".")
        start = 1
    else:
        glob_file_suffix.append(".")
        start = 0

    # for every character in string, append upper and lower case within brackets to glob file suffix list
    # then join values in list into string with no spaces
    # .CR2 (or .cr2) becomes .[Cc][Rr][22]
    for i in range(start, len(image_type)):
        up = image_type[i].upper()
        low = image_type[i].lower()
        joined = f"[{up}{low}]"
        glob_file_suffix.append(joined)

    # join list to single string
    file_suffix = ''.join(glob_file_suffix)

    # construct search name to pass to glob function
    search_name = os.path.join(config["project_path"], search_folder, "*", "*" + file_suffix)
    # return all files matching glob search parameters. return list of set to filter possible duplicates
    all_files = list(set(glob(pathname=search_name, recursive=True)))

    # check if pickle file already exists (adding more images)
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as file:
            processed_images = pkl.load(file)
            file.close()
        # remove files from all_files that are in processed_images so they are not used again
        for i in processed_images:
            del all_files[all_files.index(i)]

    # determine k to pass to random samples. If there are less images that n_images_label parameter in config file,
    # default to returning all files found in glob for labeling
    set_k = int(config["n_images_label"])
    if set_k > len(all_files):
        random_files = all_files
    else:
        # check that random choices returns files that are all truly random
        while True:
            # number of elements to pull from dict
            random_files = random.choices(all_files, k=set_k)
            if len(random_files) == len(set(random_files)):
                break
            if len(random_files) != len(set(random_files)):
                continue

    # save labeled images list as pickle. append all chosen images if it exist already
    if os.path.exists(pkl_file):
        complete_files = random_files + processed_images
        with open(pkl_file, "wb") as file:
            pkl.dump(complete_files, file)
            file.close()
    else:
        with open(pkl_file, "wb") as file:
            pkl.dump(random_files, file)
            file.close()

    return random_files


def save_csv(dictionary, save_path):
    file_name = os.path.join(save_path, "labeled_images.csv")
    # convert dict to dataframe and transpose axes
    df = pd.DataFrame(dictionary).T
    # set index to image paths
    df.index.name = "image path"
    # TODO: Could be bad if YOLOv5 allows updating the training labels for reinforcement
    # if csv exists from previous rounds of labelling, update it by reading in dataframe and concatinating
    if os.path.exists(file_name):
        current_csv = pd.read_csv(file_name, delimiter=",")
        pd.concat([current_csv, df]).to_csv(Path(file_name))
    else:
        # save as first csv
        df.to_csv(Path(file_name))


# combine csvs from subfolders
def combine_csvs(working_folder):
    search_name = os.path.join(working_folder, "*", "*" + ".csv")
    csv_files = glob(pathname=search_name, recursive=True)
    df_concat = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    return df_concat


def compare_labels(label_names, stored_labels):
    # also remove oval_id from dictionary values, leaving only tuple of coordinate points
    for i in stored_labels.keys():
        stored_labels[i] = stored_labels[i][0]
    # get keys from label_names that are not in stored_xy keys (skipped labels) and input nan value
    unlabelled_values = list(set(label_names) - set(stored_labels.keys()))
    for i in unlabelled_values:
        stored_labels[i] = np.nan
    return dict(sorted(stored_labels.items()))


class ImageLabelingToolbox:
    # TODO: assess use of config folder in class
    def __init__(self, command, output_folder=None, image_type=None, toplevel=False,
                 search_folder=None, label_names=None, thickness=10, image_size=400):

        config = yaml_utils.read_config(config)#
        project_path = config["project_path"]

        if command.__eq__("points") is False and command.__eq__("boxes") is False:
            raise ValueError

        # default thickness
        self.thickness = thickness
        # default image resize
        self.image_size = image_size

        # determine what images to use and where to store the labels based on points or boxes labeling
        project_folder = os.path.join(project_path, "bounding_boxes")
        if output_folder is None:
            self.output_folder = project_folder
        if search_folder is None:
            self.search_folder = "original_images"
        else:
            self.search_folder = os.path.basename(search_folder)
        if os.path.exists(project_folder) is False:
            os.mkdir(project_folder)
        if output_folder is not None:
            self.output_folder = output_folder

        # handle image types from previous image analysis steps
        # if called manually, use that image type
        if image_type is not None:
            self.image_type = image_type
        # if using the anything besides the original images folder, use default saved image, which is jpg
        # TODO: write default saved images type into config file
        elif self.search_folder != "original_images":
            self.image_type = ".jpg"
        # if using the original images for labeling, use the raw image type
        elif self.search_folder == "original_images":
            self.image_type = config["image_type"]
        # if all else fails, have user enter file type of images being labeled
        else:
            askstring(title="Info needed", prompt="Cannot deduce file type of images, please enter file extension. ("
                                                  "ex. jpg)")

        # establish pickle file name in case user decides to save and continue later/is continuing
        self.save_later_pkl = os.path.join(self.output_folder, "save_checkpoint.pkl")

        # determine where to load images from
        if machine_learning is True:
            self.folders_to_process = None
            self.image_list = select_images_ml(config=config, image_type=self.image_type,
                                               search_folder=self.search_folder, output_folder=self.output_folder)

        else:
            # check if the user is continuing labeling from an earlier point
            if os.path.exists(self.save_later_pkl):
                with open(self.save_later_pkl, "rb") as file:
                    self.folders_to_process = pkl.load(file)
                    file.close()

            else:
                self.folders_to_process, save_folder = directory_utils.search_existing_directories(config=config,
                                                                                               new_folder_name=os.path.basename(
                                                                                                   self.output_folder),
                                                                                               search_folder_name=self.search_folder)
            self.image_list = [os.path.join(self.folders_to_process[0], i) for i in
                               os.listdir(self.folders_to_process[0]) if
                               i.lower().endswith(self.image_type)]
            print(self.folders_to_process)
            print(self.image_list)

        # # determine if the input file is a folder or single file
        # if os.path.isdir(source_folder):
        #     self.source_folder = source_folder
        #     self.image_list = [os.path.join(source_folder, i) for i in os.listdir(source_folder) if
        #                        i.lower().endswith(self.image_type)]
        # elif os.path.isfile(source_folder):
        #     self.source_folder = os.path.dirname(source_folder)
        #     self.image_list = [source_folder]


        # determine if window wil belong to a higher level tk window
        if toplevel is False:
            self.window = tk.Tk()
        else:
            self.window = tk.Toplevel()

        # starting points for bounding box
        self.start_xy = None

        # dictionary that holds the label name: [x, y] points for each image. Is cleared with a new image
        self.stored_xy = {}

        # dictionary that holds the image name: stored_xy dictionary for each folder. Is cleared after each folder
        # is completed and saved to a csv
        self.image_points_dict = {}

        # image size to be set in canvas
        self.canvas_w = image_size
        self.canvas_h = image_size

        # set image as none to be assigned a loaded in instance in methods
        self.image = None
        # the photo assigned to the canvas
        self.photo = self.resize_image()

        # tk window is created out of multiple frames:
        # image frame is made to the size of the resized image
        self.image_frame = tk.Frame(self.window, width=self.canvas_w, height=self.canvas_h)
        # landmark frame will hold radio buttons for each landmark to be labeled
        self.landmark_frame = tk.Frame(self.window)
        # button frame holding "next" and "quit" buttons
        self.button_frame = tk.Frame(self.window)
        # Create a canvas that can fit the given image using adjusted dimensions inside image_frame
        self.canvas = tk.Canvas(self.image_frame, width=self.canvas_w, height=self.canvas_h)
        self.canvas.pack(expand=1)

        # Add PhotoImage to the Canvas
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        # bind click and crop function to mouse click
        self.canvas.bind('<Button>', self.start_bounding_box)
        self.canvas.bind('<ButtonRelease>', self.release_bounding_box)

        # create next button
        button_next = tk.Button(master=self.button_frame, text="Next", command=self.next_button)

        # create quit button
        button_quit = tk.Button(master=self.button_frame, text="Quit", command=self.window.destroy)

        # pack buttons to frame
        button_next.pack(side=tk.RIGHT)
        button_quit.pack(side=tk.LEFT)
        # TODO: add toolbar for zooming

        # pack and create the tk window
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=0)

        self.image_frame.pack(side=tk.LEFT)
        self.landmark_frame.pack(side=tk.RIGHT)
        self.button_frame.pack(side=tk.BOTTOM)
        self.window.mainloop()

    def start_bounding_box(self, event):
        # collect events
        self.start_xy = event.x, event.y

    def release_bounding_box(self, event):
        # join tuples of x,y coordinates in form upper x, upper y, lower x, lower y
        rect = self.start_xy + (event.x, event.y)
        # create rectangle and return ID
        rect_id = self.canvas.create_rectangle(rect[0], rect[1], rect[2], rect[3],
                                               width=2, outline="red")
        # stored xy for each label is list of rect points and rect ID
        self.stored_xy = [rect, rect_id]
        # reset start xy points
        self.start_xy = None

    '''function to load in image, resize it to given dimensions, and convert to a PIL image type'''

    def resize_image(self):
        # get image path from image list based on how many images have been processed
        image_path = self.image_list[len(self.image_points_dict)]
        # assign self.image to loaded in image path using openCV
        try:
            self.image = cv2.cvtColor(cv2.imread(str(Path(image_path))), cv2.COLOR_BGR2RGB)
        # error handling for a raw image
        except cv2.error:
            print("CR2 exception")
            raw = rawpy.imread(str(Path(image_path)))
            img = raw.postprocess()
            self.image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
            print(type(self.image))
        # image resolution
        r = self.image_size / self.image.shape[1]
        # image dimensions
        dim = (self.image_size, int(self.image.shape[0] * r))
        # set canvas dimensions based on image dimensions
        self.canvas_w = dim[0]
        self.canvas_h = dim[1]
        # resize cv2 image
        resized = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)
        # convert to PIL image so it can be held in tk canvas
        photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
        return photo

    '''next button to save x, y points from last image to dictionary and load next image
    If it's the last image in the folder, saves labels to a csv or xml'''

    def next_button(self):
        # get keys from label_names that are not in stored_xy keys (skipped labels) and input nan value
        # store to larger dict with image name as key, stored_xy dictionary as values
        self.image_points_dict[self.image_list[len(self.image_points_dict)]] = compare_labels(self.label_names,
                                                                                              self.stored_xy)
        # check if all images in folder have been analyzed
        if len(self.image_points_dict) == len(self.image_list):
            # save as csv
            save_csv(self.image_points_dict, self.output_folder)
            # check if labeling ML set (folders_to_process is None)
            if self.folders_to_process is not None:
                # delete first index (the folder just processed)
                del self.folders_to_process[0]
                # check if there are still folders to process
                if len(self.folders_to_process) > 0:
                    # if more folders exist to be processed, ask if user wants to continue or quit
                    value = askyesno(message="Continue labelling next folder?")
                    if value is True:
                        # if user wants to continue, reset image list
                        self.image_list = [os.path.join(self.folders_to_process[0], i) for i in
                                           os.listdir(self.folders_to_process[0]) if i.lower().endswith(self.image_type)]
                    else:
                        # user wants to quit. Save remaining folders to process and quit labeling window
                        with open(self.save_later_pkl, "wb") as file:
                            pkl.dump(self.folders_to_process, file)
                            file.close()
                        self.window.quit()
                else:
                    # no more folders to process, delete pickle file if exists
                    if os.path.exists(self.save_later_pkl):
                        os.remove(self.save_later_pkl)
                    # combine all csvs from subfolders
                    combine_df = combine_csvs(self.output_folder)
                    combine_df.to_csv(os.path.join(self.output_folder, "labeled_images.csv"))
                    self.window.quit()
            else:
                # labeling for ML and there was only one list of images generated
                self.window.quit()
        else:
            # more images to analyze within current folder
            print(self.image_points_dict)
            # resize next image
            self.photo = self.resize_image()
            # clear stored_xy points from last image
            self.stored_xy.clear()
            # clear canvas
            self.canvas.delete("all")
            # configure canvas to new dimensions
            self.canvas.configure(height=self.canvas_h, width=self.canvas_w)
            # set image to top left corner
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            # reset radio int selection variable
            self.radio_int.set(0)
