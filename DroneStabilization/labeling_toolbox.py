import cv2
import pandas as pd
from PIL import Image, ImageTk
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog
import csv
from video_functions import frame_ripper

def check_resume(videos_list, csv_file):
    csv_file = Path(csv_file)
    # if csv exist, get videos already done and subtract from video list
    if csv_file.exists():
        df = pd.read_csv(csv_file, header=0)
        processed_videos = df["video"].to_list()
        # remove already processed videos
        todo_videos = list(set(videos_list) - set(processed_videos))
        if len(todo_videos) == 0:
            raise IndexError("No videos left to label after comparision with previous maks_labels.csv file")
        return todo_videos
    else:
        return videos_list
    

'''
Take in x1,x2,y1,y2 coords and make them start from the top left corner for x and y, then add width and height
Essentially protect from an ROI selection up and to the left'''
def make_coords_safe(x_coords: tuple, y_coords: tuple):
    x = min(x_coords)
    w = max(x_coords) - x
    y = min(y_coords)
    h = max(y_coords) - y
    return x, y, w, h

  

class MaskSelectionToolbox:
    def __init__(self, video_folder, output_csv, image_size=800, toplevel=False):

        # store settings
        self.video_folder = video_folder
        self.output_csv = output_csv
        self.image_size = image_size
        self.frame_index = None

        # list of images to process
        self.video_list = [
            os.path.join(video_folder, f)
            for f in os.listdir(video_folder)
            if f.lower().endswith((".mov", ".mp4"))
        ]

        self.video_list = check_resume(self.video_list, self.output_csv)

        if len(self.video_list) == 0:
            raise ValueError("No videos found in folder!")

        # store bounding boxes: {video_path: (x1, y1, x2, y2)}
        self.labels = {}

        # determine if window wil belong to a higher level tk window
        if toplevel is False:
            self.window = tk.Tk()
        else:
            self.window = tk.Toplevel()
            
        self.window.title("Stabilization Mask Selection")

        # canvas
        self.canvas = tk.Canvas(self.window, cursor="cross")
        self.canvas.pack()

        # load first image
        self.index = 0
        self.load_image()

        # events for drawing
        self.start_xy = None
        self.rect_id = None

        self.canvas.bind("<Button-1>", self.start_box)
        self.canvas.bind("<B1-Motion>", self.draw_box)
        self.canvas.bind("<ButtonRelease-1>", self.finish_box)

        # next button
        btn_next = tk.Button(self.window, text="Next", command=self.save_and_next)
        btn_next.pack(side=tk.LEFT)

        # quit button
        btn_quit = tk.Button(self.window, text="Quit", command=self.finish)
        btn_quit.pack(side=tk.RIGHT)

        self.window.mainloop()

    # -------------------- image loading --------------------
    def load_image(self):
        path = Path(self.video_list[self.index])
        # extract middle frame and convert color
        frame_img, self.frame_index = frame_ripper(path)
        img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

        # resize keeping aspect ratio
        r = self.image_size / img.shape[1]
        dim = (self.image_size, int(img.shape[0] * r))
        img = cv2.resize(img, dim)

        self.current_image = Image.fromarray(img)
        self.tk_img = ImageTk.PhotoImage(self.current_image)

        self.canvas.config(width=dim[0], height=dim[1])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    # -------------------- bounding box drawing --------------------
    def start_box(self, event):
        self.start_xy = (event.x, event.y)

        # remove previous rectangle if any
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def draw_box(self, event):
        if self.start_xy:
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(
                self.start_xy[0], self.start_xy[1],
                event.x, event.y,
                outline="red",
                width=2
            )

    def finish_box(self, event):
        if self.start_xy:
            x, y, w, h = make_coords_safe((self.start_xy[0], event.x), (self.start_xy[1], event.y))
            self.labels[self.video_list[self.index]] = (
                x, y, w, h, self.frame_index
            )

    # -------------------- saving + next image --------------------
    def save_and_next(self):
        if self.video_list[self.index] not in self.labels:
            print("Bounding box not placed!")
            return

        # move to next image
        self.index += 1
        if self.index >= len(self.video_list):
            self.finish()
        else:
            self.load_image()

    # -------------------- finish & save CSV --------------------
    def finish(self):
        output_csv = Path(self.output_csv)
        output_csv.parent.mkdir(exist_ok=True, parents=True)
        # was mode "w", changed to "a" to support return to labelling
        with output_csv.open(mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video", "x", "y", "w", "h", "ref_frame"])
            for vid, box in self.labels.items():
                writer.writerow([vid] + list(box))

        print(f"Saved labels to {output_csv}")
        self.window.destroy()


# # -------------------- run tool --------------------
# if __name__ == "__main__":
#     folder = filedialog.askdirectory(title="Select image folder")
#     if folder:
#         MaskSelectionToolbox(folder)
