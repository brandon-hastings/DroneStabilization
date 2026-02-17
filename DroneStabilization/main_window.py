import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter import filedialog
import utils
import utils.directory_utils
import utils.yaml_utils
from segment import Segmentation, grabcut_segmentation
from labeling_toolbox import ImageLabelingToolbox
from match import IntensityMatch
from extract import LuminanceExtraction

"""Main window showing the following project steps on tk notebook tabs:
Label Images
Standardize Images
Background Segmentation
Pattern Segmentation
Extract Color Values"""


# Global variable for config file. Set by MainWindow class at initialization, then only referenced by other classes
config_file = None


class DefaultTopFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        # self.frame = tk.Frame(self)

        self.label = tk.Label(self, text="Config file")
        self.label.grid(column=0, row=0)

        self.selected_config = tk.StringVar()
        self.selected_config.set(config_file)
        self.config_entry = tk.Entry(self, textvariable=self.selected_config,
                                     width=len(utils.yaml_utils.read_config(config_file)["project_path"]), takefocus=False)
        self.config_entry.grid(column=1, row=0)

        self.config_browse = tk.Button(self, text="Browse", command=self.browse_button)
        self.config_browse.grid(column=3, row=0)

    def browse_button(self):
        filetypes = [('yaml files', '.yaml .yml')]
        global config_file
        config_file = filedialog.askopenfilename(filetypes=filetypes)
        self.selected_config.set(config_file)

        # self.frame.pack()


class DefaultBottomFrame(tk.Frame):
    def __init__(self, master, command):
        super().__init__(master)
        self.master = master
        self.command = command

        # self.frame = tk.Frame(self)

        self.button_quit = tk.Button(self, text="Quit", command=self.quit)
        self.button_quit.grid(column=0, row=0, sticky='w')

        self.button_proceed = tk.Button(self, text="Continue", command=command)
        self.button_proceed.grid(column=1, row=0, sticky='e')

        # self.frame.pack()


class LabelImages(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        self.top_frame = DefaultTopFrame(master=self)

        self.mid_frame = tk.Frame(master=self)

        self.selection_label = tk.Label(self.mid_frame, text="Select image labeling method")
        self.selection_label.grid(column=0, row=0)
        self.selected_value = tk.IntVar()
        self.bounding_box = tk.Radiobutton(self.mid_frame, text="Bounding boxes",
                                          value=0, variable=self.selected_value)
        self.bounding_box.grid(column=0, row=1)
        self.keypoint = tk.Radiobutton(self.mid_frame, text="Keypoint labeling",
                                        value=1, variable=self.selected_value)
        self.keypoint.grid(column=0, row=2)

        self.ml_label = tk.Label(self.mid_frame, text="Labeling for machine learning?")
        self.ml_label.grid(column=0, row=3)
        self.selected_value_ml = tk.IntVar()
        self.confirm = tk.Radiobutton(self.mid_frame, text="Yes",
                                           value=0, variable=self.selected_value_ml)
        self.confirm.grid(column=0, row=4, sticky="W")
        self.deny = tk.Radiobutton(self.mid_frame, text="No",
                                       value=1, variable=self.selected_value_ml)
        self.deny.grid(column=0, row=4, sticky="E")

        self.bottom_frame = DefaultBottomFrame(master=self, command=self.button_action)

        self.top_frame.grid(column=0, row=0)
        self.mid_frame.grid(column=0, row=1)
        self.bottom_frame.grid(column=0, row=2)

    def button_action(self):
        if self.selected_value_ml.get() == 0:
            ml = True
        elif self.selected_value_ml.get() == 1:
            ml = False
        else:
            # if not selected
            ml = False
        if self.selected_value.get() == 0:
            ImageLabelingToolbox(config_file, command="boxes", machine_learning=ml, toplevel=True)
        elif self.selected_value.get() == 1:
            selected_images_folder = filedialog.askdirectory(title="Select images folder to label points")
            ImageLabelingToolbox(config_file, command="points", machine_learning=ml, toplevel=True,
                                 search_folder=selected_images_folder)
        else:
            raise ValueError("No value selected for image standardization method")


class StandardizeImages(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        self.top_frame = DefaultTopFrame(master=self)

        self.mid_frame = tk.Frame(master=self)

        self.selection_label = tk.Label(self.mid_frame, text="Select image standardization method")
        self.selection_label.grid(column=0, row=0)
        self.selected_value = tk.IntVar()
        self.standardize = tk.Radiobutton(self.mid_frame, text="Standardize across images",
                                          value=0, variable=self.selected_value)
        self.standardize.grid(column=0, row=1)
        self.reference = tk.Radiobutton(self.mid_frame, text="Use first image as reference",
                                        value=1, variable=self.selected_value)
        self.reference.grid(column=0, row=2)

        self.bottom_frame = DefaultBottomFrame(master=self, command=self.button_action)

        self.top_frame.grid(column=0, row=0)
        self.mid_frame.grid(column=0, row=1)
        self.bottom_frame.grid(column=0, row=2)

    def button_action(self):
        if self.selected_value.get() == 0:
            IntensityMatch(config_file).scale_image_intensity()
        elif self.selected_value.get() == 1:
            IntensityMatch(config_file).reference_image()
        else:
            raise ValueError("No value selected for image standardization method")


class BackgroundSegment(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        self.top_frame = DefaultTopFrame(master=self)

        self.mid_frame = tk.Frame(master=self)

        self.radio_var = tk.IntVar()
        self.radio_var.set(0)
        self.choose_label = tk.Label(self.mid_frame, text="Choose segmentation method")
        self.choose_label.grid(column=0, row=0)
        self.kmeans_button = tk.Radiobutton(self.mid_frame, text="K Means", value=0, variable=self.radio_var,
                                            command=self.enable_entry)
        self.kmeans_button.grid(column=0, row=1)
        self.auto_grabcut = tk.Radiobutton(self.mid_frame, text="Automatic grabCut", value=1, variable=self.radio_var,
                                           command=self.disable_entry)
        self.auto_grabcut.grid(column=0, row=2)
        self.man_grabcut = tk.Radiobutton(self.mid_frame, text="Manual grabCut", value=2, variable=self.radio_var,
                                          command=self.disable_entry)
        self.man_grabcut.grid(column=0, row=3)

        self.cluster_label = tk.Label(self.mid_frame, text="Number of clusters:")
        self.cluster_label.grid(column=0, row=4)
        self.cluster_var = tk.StringVar()
        self.cluster_entry = tk.Entry(self.mid_frame, textvariable=self.cluster_var, takefocus=True)
        self.cluster_entry.grid(column=1, row=4)

        self.bottom_frame = DefaultBottomFrame(master=self, command=self.button_action)

        self.top_frame.grid(column=0, row=0)
        self.mid_frame.grid(column=0, row=1)
        self.bottom_frame.grid(column=0, row=2)

    def enable_entry(self):
        self.cluster_label.configure(state='normal')
        self.cluster_entry.configure(state='normal')

    def disable_entry(self):
        self.cluster_label.configure(state='disabled')
        self.cluster_entry.configure(state='disabled')

    def button_action(self):
        if self.radio_var == 0:
            cluster = self.cluster_var.get()

            if cluster is not None:

                try:
                    int_cluster = int(cluster)
                    Segmentation(config_file, n_cluster=int_cluster).background_segmentation()
                except ValueError:
                    print("value for number of clusters is {}, must be an integer.".format(cluster))
            else:
                raise TypeError("No value input for number of clusters")

        elif self.radio_var == 1:
            grabcut_segmentation(config_file)



class SegmentPattern(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        self.top_frame = DefaultTopFrame(master=self)

        self.mid_frame = tk.Frame(master=self)
        # Task specific widgets
        self.selection_label = tk.Label(self.mid_frame, text="Select color pattern segmentation method")
        self.selection_label.grid(column=0, row=0)
        self.selected_value = tk.IntVar()
        self.threshold = tk.Radiobutton(self.mid_frame, text="Threshold values (automatically extracted)",
                                        value=0, variable=self.selected_value, command=self.disable_entry)
        self.threshold.grid(column=0, row=1)
        self.kmeans = tk.Radiobutton(self.mid_frame, text="kMeans segmentation (with user selection)",
                                     value=1, variable=self.selected_value, command=self.enable_entry)
        self.kmeans.grid(column=0, row=2)

        self.cluster_label = tk.Label(self.mid_frame, text="Number of clusters:", state='disabled')
        self.cluster_label.grid(column=0, row=3)
        self.cluster_var = tk.StringVar()
        self.cluster_entry = tk.Entry(self.mid_frame, textvariable=self.cluster_var, state='disabled')
        self.cluster_entry.grid(column=1, row=3)

        self.bottom_frame = DefaultBottomFrame(master=self, command=self.button_action)

        self.top_frame.grid(column=0, row=0)
        self.mid_frame.grid(column=0, row=1)
        self.bottom_frame.grid(column=0, row=2)

    def enable_entry(self):
        self.cluster_label.configure(state='normal')
        self.cluster_entry.configure(state='normal')

    def disable_entry(self):
        self.cluster_label.configure(state='disabled')
        self.cluster_entry.configure(state='disabled')

    def button_action(self):
        if self.selected_value.get() == 0:
            LuminanceExtraction(config_file).automatic_color_segmentation()

        elif self.selected_value.get() == 1:
            cluster = self.cluster_var.get()

            if cluster is not None:

                try:
                    int_cluster = int(cluster)
                    Segmentation(config_file, n_cluster=int_cluster).manual_pattern_segmentation()
                except ValueError:
                    print("value for number of clusters is {}, must be an integer.".format(cluster))
            # else cluster variable is None, i.e. not input by the user
            else:
                raise TypeError("No value input for number of clusters")


class ExtractValues(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        self.top_frame = DefaultTopFrame(master=self)

        self.mid_frame = tk.Frame(master=self)
        # Task specific widgets
        self.extract_label = tk.Label(master=self.mid_frame, text="Extract manually segmented images")
        self.extract_label.grid(column=0, row=0)

        self.bottom_frame = DefaultBottomFrame(master=self, command=self.button_action)

        self.top_frame.grid(column=0, row=0)
        self.mid_frame.grid(column=0, row=1)
        self.bottom_frame.grid(column=0, row=2)

    @staticmethod
    def button_action():
        LuminanceExtraction(config_file).extract_manual_segmentations()


class Notebook(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.container = container
        self.notebook = ttk.Notebook(container)
        self.add_tab(LabelImages(container), "Label Images")
        self.add_tab(StandardizeImages(container), "Image Standardization")
        self.add_tab(BackgroundSegment(container), "Background Segmentation")
        self.add_tab(SegmentPattern(container), "Pattern Segmentation")
        self.add_tab(ExtractValues(container), "Extract color values")

    def add_tab(self, frame, title):
        self.notebook.add(frame, text=title)
        self.notebook.pack()
        # self.notebook.grid(column=0, row=0)


class MainWindow(tk.Tk):
    def __init__(self, config):
        super().__init__()
        self.title("Lumeleon")
        # self.geometry('500x500')
        self.resizable(True, True)

        global config_file
        config_file = config


def main(config):
    main_window = MainWindow(config)
    Notebook(main_window)
    main_window.mainloop()


'''Not intended to be called from command line, but can be done with config file path as argument'''
if __name__ == "__main__":
    main(Path(sys.argv[1]))
