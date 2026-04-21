from tkinter import filedialog
from PIL import ImageTk, ImageOps
import numpy as np
import colorsys

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

def isColorful(rgb, sat_threshold=0.2):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return s > sat_threshold

