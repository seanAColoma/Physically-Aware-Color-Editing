import numpy as np
import tkinter as tk
import utils
from tkinter import colorchooser

class Layer:
    def __init__(self, image, color):
        self.image = image
        self.originalColor = color
        self.color = color

    def updateColor(self, newColor):
        offset = (newColor.astype(np.float32) - self.color.astype(np.float32))
        self.color = newColor

        img = self.image[:, :, :3].astype(np.float32)
        alpha = self.image[:, :, 3]
        mask = (alpha > 0)

        img[mask] = img[mask] + offset

        self.image[:, :, :3] = np.clip(img, 0, 255).astype(np.uint8)

    def resetColor(self):
        self.updateColor(self.originalColor)
