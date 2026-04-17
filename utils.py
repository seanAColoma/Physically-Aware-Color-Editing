from tkinter import filedialog
from PIL import ImageTk, ImageOps
import numpy as np

def selectImageFile():
    filePath = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    return filePath

def selectTextFile():
    filePath = filedialog.askopenfilename(
        filetypes=[("Text files", "*.txt")]
    )
    return filePath


def rgbToHex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'

def readRGBFromText(rgbFilePath):
    rgb = np.loadtxt(rgbFilePath, delimiter=',', dtype=np.uint8)
    return rgb

def recontructFromLayers(albedo, diffuse, residual):
    print("reconstucting...")

def getTkinterImage(pilImage):
    return ImageTk.PhotoImage(ImageOps.contain(pilImage, (512,512)))