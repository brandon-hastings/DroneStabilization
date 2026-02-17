import cv2
from PIL import Image, ImageTk
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog
import csv


def frame_ripper(video_path):
    '''
    Docstring for frame_ripper
    
    :param video_path: path to video

    :retval frame: middle frame of video for creating a bounding box
    '''
    # where video is expected to be the filename of a video
    source = cv2.VideoCapture(str(video_path))
    frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = int(frames/2)
    source.set(cv2.CAP_PROP_POS_FRAMES, middle_frame-1)
    res, frame = source.read()
    if not res:
        IndexError("No frame found")
    else:
        return frame


class MaskSelectionToolbox:
    def __init__(self, video_folder, output_csv="labels.csv", image_size=800):

        # store settings
        self.video_folder = video_folder
        self.output_csv = output_csv
        self.image_size = image_size

        # list of images to process
        self.video_list = [
            os.path.join(video_folder, f)
            for f in os.listdir(video_folder)
            if f.lower().endswith((".mov", ".mp4"))
        ]

        if len(self.video_list) == 0:
            raise ValueError("No videos found in folder!")

        # store bounding boxes: {video_path: (x1, y1, x2, y2)}
        self.labels = {}

        # Tkinter window
        self.window = tk.Tk()
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
        path = self.video_list[self.index]
        # extract middle frame and convert color
        img = cv2.cvtColor(frame_ripper(path), cv2.COLOR_BGR2RGB)

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
            self.labels[self.video_list[self.index]] = (
                self.start_xy[0], self.start_xy[1],
                event.x, event.y,
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
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video", "x1", "y1", "x2", "y2"])
            for vid, box in self.labels.items():
                writer.writerow([os.path.basename(vid)] + list(box))

        print(f"Saved labels to {self.output_csv}")
        self.window.destroy()


# -------------------- run tool --------------------
if __name__ == "__main__":
    folder = filedialog.askdirectory(title="Select image folder")
    if folder:
        SimpleBoundingBoxTool(folder)
