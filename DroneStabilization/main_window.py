import sys
import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter import filedialog
import utils
import utils.yaml_utils
from labeling_toolbox import MaskSelectionToolbox
from video_functions import batch_stabilize

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


class MaskSelection(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        self.top_frame = DefaultTopFrame(master=self)

        self.mid_frame = tk.Frame(master=self)
        
        self.selection_label = tk.Label(self.mid_frame, text="Path to take videos from")
        self.selection_label.grid(column=0, row=0)
        self.default_path = os.path.join(utils.yaml_utils.read_config(config_file)["project_path"],
                                              "original_data")
        self.selected_path = tk.StringVar()
        self.selected_path.set(self.default_path)
        self.path_entry = tk.Entry(self.mid_frame, textvariable=self.default_path,
                                     width=len(self.default_path), takefocus=False)
        self.path_entry.grid(column=1, row=0)

        self.config_browse = tk.Button(self.mid_frame, text="Browse", command=self.browse_button)
        self.config_browse.grid(column=3, row=0)

        self.bottom_frame = DefaultBottomFrame(master=self, command=self.button_action)

        self.top_frame.grid(column=0, row=0)
        self.mid_frame.grid(column=0, row=1)
        self.bottom_frame.grid(column=0, row=2)

    def browse_button(self):
        selected_folder = filedialog.askdirectory(initialdir=Path(self.default_path).parent)
        self.selected_path.set(selected_folder)
        self.default_path = selected_folder

    def button_action(self):
        MaskSelectionToolbox(self.default_path,
                             output_csv=Path(self.default_path).parents[1] / "results" / "mask_labels.csv",
                             toplevel=True)


class StabilizeVideos(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        self.top_frame = DefaultTopFrame(master=self)

        self.mid_frame = tk.Frame(master=self)

        self.selection_label = tk.Label(self.mid_frame, text="Path to stabilize videos from")
        self.selection_label.grid(column=0, row=0)
        self.default_path = os.path.join(utils.yaml_utils.read_config(config_file)["project_path"],
                                              "original_data")
        self.selected_path = tk.StringVar()
        self.selected_path.set(self.default_path)
        self.path_entry = tk.Entry(self.mid_frame, textvariable=self.default_path,
                                     width=len(self.default_path), takefocus=False)
        self.path_entry.grid(column=1, row=0)

        self.config_browse = tk.Button(self.mid_frame, text="Browse", command=self.browse_button)
        self.config_browse.grid(column=3, row=0)

        self.bottom_frame = DefaultBottomFrame(master=self, command=self.button_action)

        self.top_frame.grid(column=0, row=0)
        self.mid_frame.grid(column=0, row=1)
        self.bottom_frame.grid(column=0, row=2)

    def browse_button(self):
        selected_folder = filedialog.askdirectory(initialdir=Path(self.default_path).parent)
        self.selected_path.set(selected_folder)
        self.default_path = selected_folder

    def button_action(self):
        batch_stabilize(self.default_path,
                        mask_csv=Path(self.default_path).parents[1] / "results" / "mask_labels.csv",
                        shifts_csv=Path(self.default_path).parents[1] / "results" / "dxdy_shifts.csv")


class Notebook(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.container = container
        self.notebook = ttk.Notebook(container)
        self.add_tab(MaskSelection(container), "Mask Selection")
        self.add_tab(StabilizeVideos(container), "Video Stabilization")

    def add_tab(self, frame, title):
        self.notebook.add(frame, text=title)
        self.notebook.pack()
        # self.notebook.grid(column=0, row=0)


class MainWindow(tk.Tk):
    def __init__(self, config):
        super().__init__()
        self.title("DroneStabilization")
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
