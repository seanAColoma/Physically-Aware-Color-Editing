import numpy as np
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter import colorchooser
import colorEditing 
import utils
from layer import Layer


'''
notes:

consider organizing with classes

save color values as txt or csv files (whichever is easier to read later) (ask Duc to modify the c++ segmentation code for this)

relation between albedo palette colors/layers and segmented shading...

save previous palette and edited palette? Probably can be done when the shading code is called...

np images for everything except display elements i.e. only display image should be a PIL image - convert when display should be updated

button to do decomposition

extend cpp segmentation code to output rgb values per layer
'''

window = tk.Tk()
layerColorButtonFrame = tk.Frame()
notebook = ttk.Notebook(window)
notebook.pack(side="left",expand=True)


alphaLayers = []
albedoLayers = []
layerColorButtons = []
layerColorValues = []

# consider renaming these
albedoNumpy = np.zeros((512, 512, 3), dtype=np.uint8)
shadingNumpy = np.zeros((512, 512, 3), dtype=np.uint8)
residualNumpy = np.zeros((512, 512, 3), dtype=np.uint8)
resultNumpy = np.zeros((512, 512), dtype=np.uint8)

albedoFrame = tk.Frame(notebook)
shadingFrame = tk.Frame(notebook)
editedShading = tk.Frame(notebook)
residualFrame = tk.Frame(notebook)
editedResidual = tk.Frame(notebook)
resultFrame = tk.Frame(notebook)

notebook.add(albedoFrame, text="Albedo")
notebook.add(shadingFrame, text="Shading")
# notebook.add(editedShading, text="Edited Shading")
notebook.add(residualFrame, text="Residual")
# notebook.add(editedResidual, text="Edited Residual")
notebook.add(resultFrame, text="Result")

rightFrame = tk.Frame(window)

# update
displayAlbedoImage = ImageTk.PhotoImage(Image.fromarray(albedoNumpy))
displayAlbedoImageLabel = tk.Label(master=albedoFrame, image = displayAlbedoImage)
displayAlbedoImageLabel.image = displayAlbedoImage
displayAlbedoImageLabel.pack()

displayDiffuseShadingImage = ImageTk.PhotoImage(Image.fromarray(shadingNumpy))
displayDiffuseShadingImageLabel = tk.Label(master=shadingFrame, image = displayDiffuseShadingImage)
displayDiffuseShadingImageLabel.image = displayDiffuseShadingImage
displayDiffuseShadingImageLabel.pack()

displayResidualImage = ImageTk.PhotoImage(Image.fromarray(residualNumpy))
displayResidualImageLabel = tk.Label(master=residualFrame, image = displayResidualImage)
displayResidualImageLabel.image = displayResidualImage
displayResidualImageLabel.pack()

displayResultImage = ImageTk.PhotoImage(Image.fromarray(resultNumpy))
displayResultImageLabel = tk.Label(master=resultFrame, image = displayResultImage)
displayResultImageLabel.image = displayResultImage
displayResultImageLabel.pack()

def addAlbedoLayer():
    global albedoNumpy
    global displayAlbedoImage
    global albedoLayers

    filePath = utils.selectImageFile()
    rgbFilePath = utils.selectTextFile()
    
    selectedLayer = Image.open(filePath)
    rgbValue = utils.readRGBFromText(rgbFilePath)

    numpyLayer = np.asarray(selectedLayer).copy()

    newLayer = Layer(numpyLayer, rgbValue)
    albedoLayers.append(newLayer)
    
    newButton = createNewLayerColorButton(rgbValue, newLayer)
    layerColorButtons.append(newButton)

    updateButtons(layerColorButtons)
    
    albedoNumpy = updateCompositing(albedoLayers)
    updateAlbedo()

def addShading():
    global shadingNumpy
    filePath = utils.selectImageFile()
    selectedImage = Image.open(filePath)
    shadingNumpy = np.asarray(selectedImage)

    updateShading()

def addResidual():
    global residualNumpy
    filePath = utils.selectImageFile()
    selectedImage = Image.open(filePath)
    residualNumpy = np.asarray(selectedImage)

    updateResidual()
def createNewLayerColorButton(color, layer):
    global displayAlbedoImageLabel
    hexColor = utils.rgbToHex(color[0], color[1], color[2])
    newButton = tk.Button(
        background= hexColor,
        height=1,
        width=2,
        padx= 5,
        pady= 5
    )

    newButton.config(command=lambda b=newButton, l=layer: chooseColor(b, l))

    return newButton

def updateButtons(buttons):
    for button in buttons:
        button.pack()

def chooseColor(button, layer):
    color = colorchooser.askcolor(title="Choose A Color")
    
    if color[1]:
        button.config(bg=color[1])
    if color[0]:
        layer.updateColor(np.asarray(color[0]).copy())
    
    updateAlbedo()
    
def updateCompositing(alphaLayers):
    return colorEditing.reconstructFromLayers(alphaLayers)

def updateAlbedo():
    global displayAlbedoImage
    global displayAlbedoImageLabel
    global albedoNumpy
    global albedoLayers

    albedoNumpy = updateCompositing(albedoLayers)

    displayAlbedoImage = Image.fromarray(albedoNumpy)
    displayImageTk = utils.getTkinterImage(displayAlbedoImage)
    displayAlbedoImageLabel.config(image = displayImageTk)
    displayAlbedoImageLabel.image = displayImageTk

def updateShading():
    global displayDiffuseShadingImage
    global displayDiffuseShadingImageLabel
    global shadingNumpy

    displayDiffuseShadingImage = Image.fromarray(shadingNumpy)
    displayImageTk = utils.getTkinterImage(displayDiffuseShadingImage)
    displayDiffuseShadingImageLabel.config(image = displayImageTk)
    displayDiffuseShadingImageLabel.image = displayImageTk

def updateResidual():
    global displayResidualImage
    global displayResidualImageLabel
    global residualNumpy

    displayResidualImage = Image.fromarray(residualNumpy)
    displayImageTk = utils.getTkinterImage(displayResidualImage)
    displayResidualImageLabel.config(image = displayImageTk)
    displayResidualImage.image = displayImageTk


# eventually update this to save the actual result image
def saveImageCommand(image, name):
    pilResult = Image.fromarray(image)
    pilResult.save(name)

def resetLayerColorsCommand(layers):
    for layer in layers:
        layer.resetColor()

addAlbedoLayerButton = tk.Button(
    text = "Add Albedo Color Layer",
    command=addAlbedoLayer,
    master = albedoFrame 
)

addAlbedoLayerButton.pack(side="bottom")

# should this be on a segment basis?
addShadingButton = tk.Button(
    text = "Add Shading",
    master = shadingFrame,
    command=addShading
)
addShadingButton.pack(side="bottom")

# should this be on a segment basis?
addResidualButton = tk.Button(
    text = "Add Residual",
    master = residualFrame,
    command=addResidual
)
addResidualButton.pack(side="bottom")

saveAlbedoImageButton = tk.Button(
    text = "Save Albedo Image",
    command=lambda: saveImageCommand(albedoNumpy, "albedo.png"),
    master = albedoFrame
)

saveShadingImageButton = tk.Button(
    text = "Save Shading",
    command=lambda: saveImageCommand(shadingNumpy, "shading.png"),
    master = shadingFrame
)

saveResidualImageButton = tk.Button(
    text = "Save Residual",
    command=lambda: saveImageCommand(residualNumpy, "residual.png"),
    master = residualFrame
)

saveResultImageButton = tk.Button(
    text = "Save Result",
    command=lambda: saveImageCommand(resultNumpy, "result.png"),
    master = resultFrame
)

saveResultImageButton.pack(side="bottom")
saveResidualImageButton.pack(side="bottom")
saveShadingImageButton.pack(side="bottom")
saveAlbedoImageButton.pack(side="bottom")

rightFrame.pack(side="right")
layerColorButtonFrame.pack()


window.mainloop()