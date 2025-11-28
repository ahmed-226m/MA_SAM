import tkinter as tk
from tkinter import filedialog
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NiiViewer:
    def __init__(self, master):
        self.master = master
        master.title("NIfTI Image Viewer")

        self.fig, self.ax = plt.subplots(1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.slice_slider = tk.Scale(master, from_=0, to=0, orient=tk.HORIZONTAL, command=self.update_slice)
        self.slice_slider.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_open = tk.Button(master, text="Open .nii File", command=self.open_file)
        self.btn_open.pack(side=tk.BOTTOM)

        self.data = None

    def open_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii;*.nii.gz"), ("All files", "*.*")])
        if not filepath:
            return
        
        self.load_nii(filepath)

    def load_nii(self, filepath):
        try:
            img = nib.load(filepath)
            self.data = img.get_fdata()

            if self.data.ndim == 4:
                self.data = self.data[:, :, :, 0]
            
            if self.data.ndim != 3:
                print(f"Error: Loaded data is not a 3D image. Dimensions: {self.data.ndim}")
                self.data = None
                return

            self.slice_slider.config(from_=0, to=self.data.shape[1] - 1)
            self.slice_slider.set(self.data.shape[1] // 2)
            self.update_slice(self.slice_slider.get())
        except Exception as e:
            print(f"Error loading NIfTI file: {e}")

    def update_slice(self, slice_index):
        if self.data is not None:
            slice_index = int(slice_index)
            self.ax.clear()
            self.ax.imshow(np.rot90(self.data[:, slice_index, :]), cmap='gray')
            self.ax.axis('off')
            self.canvas.draw()

def main():
    root = tk.Tk()
    app = NiiViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
