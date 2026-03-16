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
from roi_evaluation import batch_suggest_rois

"""Main window showing the following project steps on tk notebook tabs:
ROI selection
Calculate shifts"""


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
        self.selected_path = tk.StringVar(value=self.default_path)
        self.path_entry = tk.Entry(self.mid_frame, textvariable=self.selected_path,
                                     width=len(self.default_path), takefocus=False)
        self.path_entry.grid(column=1, row=0)

        self.config_browse = tk.Button(self.mid_frame, text="Browse", command=self.browse_button)
        self.config_browse.grid(column=2, row=0)

        self.radio_var = tk.IntVar()
        self.radio_var.set(1)
        self.choose_label = tk.Label(self.mid_frame, text="Choose ROI selection method")
        self.choose_label.grid(column=0, row=1)

        self.automatic_button = tk.Radiobutton(self.mid_frame, text="Automatic", value=1, variable=self.radio_var,
                                           command=self.enable_entry)
        self.automatic_button.grid(column=0, row=2)
        self.manual_button = tk.Radiobutton(self.mid_frame, text="Manual", value=0, variable=self.radio_var,
                                            command=self.disable_entry)
        self.manual_button.grid(column=0, row=3)

        self.size_label = tk.Label(self.mid_frame, text="Region size (square):")
        self.size_label.grid(column=1, row=1)
        self.default_size = "256"
        self.size_var = tk.StringVar(value=self.default_size)
        self.size_entry = tk.Entry(self.mid_frame, textvariable=self.size_var, takefocus=False)
        self.size_entry.grid(column=1, row=2)

        self.frames_label = tk.Label(self.mid_frame, text="Frames to sample:")
        self.frames_label.grid(column=2, row=1)
        self.default_frames = "20"
        self.frames_var = tk.StringVar(value=self.default_frames)
        self.frames_entry = tk.Entry(self.mid_frame, textvariable=self.frames_var, takefocus=False)
        self.frames_entry.grid(column=2, row=2)

        self.regions_label = tk.Label(self.mid_frame, text="Regions to rank:")
        self.regions_label.grid(column=3, row=1)
        self.default_regions = "3"
        self.regions_var = tk.StringVar(value=self.default_regions)
        self.regions_entry = tk.Entry(self.mid_frame, textvariable=self.regions_var, takefocus=False)
        self.regions_entry.grid(column=3, row=2)

        self.bottom_frame = DefaultBottomFrame(master=self, command=self.button_action)

        self.top_frame.grid(column=0, row=0)
        self.mid_frame.grid(column=0, row=1)
        self.bottom_frame.grid(column=0, row=2)

    def enable_entry(self):
        self.size_label.configure(state='normal')
        self.size_entry.configure(state='normal')
        self.frames_label.configure(state='normal')
        self.frames_entry.configure(state='normal')
        self.regions_label.configure(state='normal')
        self.regions_entry.configure(state='normal')


    def disable_entry(self):
        self.size_label.configure(state='disabled')
        self.size_entry.configure(state='disabled')
        self.frames_label.configure(state='disabled')
        self.frames_entry.configure(state='disabled')
        self.regions_label.configure(state='disabled')
        self.regions_entry.configure(state='disabled')

    def browse_button(self):
        selected_folder = filedialog.askdirectory(initialdir=Path(self.default_path).parent)
        self.selected_path.set(selected_folder)

    def button_action(self):
        if self.radio_var.get() == 0:
            print("opening toolbox")
            MaskSelectionToolbox(self.selected_path.get(),
                                output_csv=Path(self.default_path).parent / "results" / "mask_labels.csv",
                                toplevel=True)
        elif self.radio_var.get() == 1:
            print("running suggest rois")
            batch_suggest_rois(self.selected_path.get(),
                               output_csv=Path(self.default_path).parent / "results" / "mask_labels.csv",
                               roi_size=(256, 256),
                               k=3,
                               samples=20
                               )
        else:
            print("Error in radio button selection")
            



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
        self.selected_path = tk.StringVar(value=self.default_path)
        self.path_entry = tk.Entry(self.mid_frame, textvariable=self.selected_path,
                                     width=len(self.default_path), takefocus=False)
        self.path_entry.grid(column=1, row=0)

        self.config_browse = tk.Button(self.mid_frame, text="Browse", command=self.browse_button)
        self.config_browse.grid(column=3, row=0)

        self.radio_var = tk.IntVar()
        self.radio_var.set(0)
        self.choose_label = tk.Label(self.mid_frame, text="Choose transformation method")
        self.choose_label.grid(column=0, row=1)

        self.automatic_button = tk.Radiobutton(self.mid_frame, text="Translation", value=0, variable=self.radio_var)
        self.automatic_button.grid(column=0, row=2)
        self.manual_button = tk.Radiobutton(self.mid_frame, text="Affine transformation", value=1, variable=self.radio_var)
        self.manual_button.grid(column=0, row=3)

        self.bottom_frame = DefaultBottomFrame(master=self, command=self.button_action)

        self.top_frame.grid(column=0, row=0)
        self.mid_frame.grid(column=0, row=1)
        self.bottom_frame.grid(column=0, row=2)

    def browse_button(self):
        selected_folder = filedialog.askdirectory(initialdir=Path(self.default_path).parent)
        self.selected_path.set(selected_folder)

    def button_action(self):
        if self.radio_var.get() == 0:
            print("Running translation based stabilization")
            batch_stabilize(self.selected_path.get(),
                        mask_csv=Path(self.default_path).parent / "results" / "mask_labels.csv",
                        shifts_csv=Path(self.default_path).parent / "results" / "dxdy_shifts.csv",
                        method="translation")
        elif self.radio_var.get() == 1:
            print("Running affine based stabilization")
            batch_stabilize(self.selected_path.get(),
                        mask_csv=Path(self.default_path).parent / "results" / "mask_labels.csv",
                        shifts_csv=Path(self.default_path).parent / "results" / "dxdy_shifts.csv",
                        method="affine")


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
