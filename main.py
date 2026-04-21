import numpy as np
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter import colorchooser
import colorEditing 
import utils
from albedoLayer import Layer
from selectableLayer import SelectableLayer
from editShadingRGBSegments import solveShadingRGB, editShadingRGBSegments, combineSegments, editSingleShadingRGBSegment
import torch
from function.reconstruct import perform_recolor

window = tk.Tk()
layerColorButtonFrame = tk.Frame()
notebook = ttk.Notebook(window)
notebook.pack(side="left",expand=True)

albedoLayers = []
shadingLayers = []
residualLayers = []

layerColorButtons = []
shadingTickboxes = []
residualTickboxes = []
participatingLayers = []

shadingDecomp = torch.zeros(3, 4) 
residualDecomp = torch.zeros(3, 4)

albedoNumpy = np.zeros((512, 512, 3), dtype=np.uint8)
shadingNumpy = np.zeros((512, 512, 3), dtype=np.uint8)
selectedShadingNumpy = np.zeros((512, 512, 3), dtype=np.uint8)

residualNumpy = np.zeros((512, 512, 3), dtype=np.uint8)
selectedResidualNumpy = np.zeros((512, 512, 3), dtype=np.uint8)

resultNumpy = np.zeros((512, 512), dtype=np.uint8)

albedoFrame = tk.Frame(notebook)

shadingFrame = tk.Frame(notebook)
selectedShadingFrame = tk.Frame(notebook)

residualFrame = tk.Frame(notebook)
selectedResidualFrame = tk.Frame(notebook)

resultFrame = tk.Frame(notebook)

notebook.add(albedoFrame, text="Albedo")
notebook.add(shadingFrame, text="Shading")
notebook.add(selectedShadingFrame, text="Selected Shading")
notebook.add(residualFrame, text="Residual")
notebook.add(selectedResidualFrame, text="Edited Residual")
notebook.add(resultFrame, text="Result")

rightFrame = tk.Frame(window)

displayAlbedoImage = ImageTk.PhotoImage(Image.fromarray(albedoNumpy))
displayAlbedoImageLabel = tk.Label(master=albedoFrame, image = displayAlbedoImage)
displayAlbedoImageLabel.image = displayAlbedoImage
displayAlbedoImageLabel.pack()

displayDiffuseShadingImage = ImageTk.PhotoImage(Image.fromarray(shadingNumpy))
displayDiffuseShadingImageLabel = tk.Label(master=shadingFrame, image = displayDiffuseShadingImage)
displayDiffuseShadingImageLabel.image = displayDiffuseShadingImage
displayDiffuseShadingImageLabel.pack()

displaySelectedShadingImage = ImageTk.PhotoImage(Image.fromarray(selectedShadingNumpy))
displaySelectedShadingImageLabel = tk.Label(master=selectedShadingFrame, image = displaySelectedShadingImage)
displaySelectedShadingImageLabel.image = displaySelectedShadingImage
displaySelectedShadingImageLabel.pack()


displayResidualImage = ImageTk.PhotoImage(Image.fromarray(residualNumpy))
displayResidualImageLabel = tk.Label(master=residualFrame, image = displayResidualImage)
displayResidualImageLabel.image = displayResidualImage
displayResidualImageLabel.pack()

displaySelectedResidualImage = ImageTk.PhotoImage(Image.fromarray(selectedResidualNumpy))
displaySelectedResidualImageLabel = tk.Label(master=selectedResidualFrame, image = displaySelectedResidualImage)
displaySelectedResidualImageLabel.image = displaySelectedResidualImage
displaySelectedResidualImageLabel.pack()

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
    
    selectedLayer = Image.open(filePath).convert("RGBA")
    rgbValue = utils.readRGBFromText(rgbFilePath)

    numpyLayer = np.asarray(selectedLayer).copy()

    newLayer = Layer(numpyLayer, rgbValue)
    albedoLayers.append(newLayer)
    
    newButton = createNewLayerColorButton(rgbValue, newLayer)
    layerColorButtons.append(newButton)

    updateButtons(layerColorButtons)
    
    imageList = []

    for layer in albedoLayers:
        imageList.append(layer.image)
    albedoNumpy = updateCompositing(imageList)
    updateAlbedo()

def addShadingLayer():
    global shadingNumpy
    global displayDiffuseShadingImage
    global shadingLayers

    filePath = utils.selectImageFile()
    selectedLayer = Image.open(filePath).convert("RGBA")

    numpyLayer = np.asarray(selectedLayer).copy()

    tkBool = tk.BooleanVar()
    newLayer = SelectableLayer(numpyLayer, tkBool, len(shadingLayers))

    shadingLayers.append(newLayer)

    imageList = []
    for layer in shadingLayers:
        imageList.append(layer.image)

    tickbox = createNewSelectedCheckBox(selectedShadingFrame, newLayer, shadingLayers, updateSelectedShading)
    shadingTickboxes.append(tickbox)

    shadingNumpy = updateCompositing(imageList)
    updateShading()
    updateTickboxes(shadingTickboxes)

def addResidualLayer():
    global residualNumpy
    global residualLayers

    filePath = utils.selectImageFile()
    selectedLayer = Image.open(filePath).convert("RGBA")

    numpyLayer = np.asarray(selectedLayer).copy()

    tkBool = tk.BooleanVar()
    newLayer = SelectableLayer(numpyLayer, tkBool, len(residualLayers))
    residualLayers.append(newLayer)

    imageList = []
    for layer in residualLayers:
        imageList.append(layer.image)

    tickbox = createNewSelectedCheckBox(selectedResidualFrame, newLayer, residualLayers, updateSelectedResidual)
    residualTickboxes.append(tickbox)

    residualNumpy = updateCompositing(imageList)

    updateResidual()
    updateTickboxes(residualTickboxes)

def createNewLayerColorButton(color, layer):
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

def createNewSelectedCheckBox(master, layer, allLayers, updateFunction):

    tickbox = tk.Checkbutton(
        master, 
        text="Select Layer: " + str(layer.index),
        variable=layer.selected,
        command=lambda s = allLayers: updateFunction(s))
    return tickbox

def updateTickboxes(tickboxes, side = "right"):
    for tickbox in tickboxes:
        tickbox.pack(side=side)

def updateButtons(buttons):
    for button in buttons:
        button.pack()

def chooseColor(button, layer):
    global resultNumpy

    color = colorchooser.askcolor(title="Choose A Color")
    
    if color[1]:
        button.config(bg=color[1])
    if color[0]:
        layer.updateColor(np.asarray(color[0]).copy())
    
    updateAlbedo()
    applyDecompositionToShading()
    applyDecompositionToResidual()

    resultNumpy = reconstructImage()

    updateShading()
    updateResidual()
    updateResult()


def updateShadingSelections(shadingLayers):
    updateSelectedShading(shadingLayers)

def updateCompositing(alphaLayers):
    return colorEditing.reconstructFromLayers(alphaLayers)

def updateAlbedo():
    global displayAlbedoImage
    global displayAlbedoImageLabel
    global albedoNumpy
    global albedoLayers

    imageList = []
    for layer in albedoLayers:
        imageList.append(layer.image)
    albedoNumpy = updateCompositing(imageList)

    displayAlbedoImage = Image.fromarray(albedoNumpy)
    displayImageTk = utils.getTkinterImage(displayAlbedoImage)
    displayAlbedoImageLabel.config(image = displayImageTk)
    displayAlbedoImageLabel.image = displayImageTk

def updateSelectedShading(layers):
    global selectedShadingNumpy
    global displaySelectedShadingImage

    print("update shading selection")

    selectedLayers = []

    for layer in layers:
        if(layer.selected.get()):
            selectedLayers.append(layer.image)

    selectedShadingNumpy = updateCompositing(selectedLayers)
    displaySelectedShadingImage = Image.fromarray(selectedShadingNumpy)
    displayImageTk = utils.getTkinterImage(displaySelectedShadingImage)
    displaySelectedShadingImageLabel.config(image = displayImageTk)
    displaySelectedShadingImageLabel.image = displayImageTk

# update the display image
def updateShading():
    global displayDiffuseShadingImage
    global displayDiffuseShadingImageLabel
    global shadingNumpy

    displayDiffuseShadingImage = Image.fromarray(shadingNumpy)
    displayImageTk = utils.getTkinterImage(displayDiffuseShadingImage)
    displayDiffuseShadingImageLabel.config(image = displayImageTk)
    displayDiffuseShadingImageLabel.image = displayImageTk

def updateSelectedResidual(layers):
    global selectedResidualNumpy
    global displaySelectedResidualImage
    selectedLayers = []

    print("update residual selection")

    for layer in layers:
        if(layer.selected.get()):
            selectedLayers.append(layer.image)

    selectedResidualNumpy = updateCompositing(selectedLayers)
    displaySelectedResidualImage = Image.fromarray(selectedResidualNumpy)
    displayImageTk = utils.getTkinterImage(displaySelectedResidualImage)
    displaySelectedResidualImageLabel.config(image = displayImageTk)
    displaySelectedResidualImageLabel.image = displayImageTk

def updateResidual():
    global displayResidualImage
    global displayResidualImageLabel
    global residualNumpy

    displayResidualImage = Image.fromarray(residualNumpy)
    displayImageTk = utils.getTkinterImage(displayResidualImage)
    displayResidualImageLabel.config(image = displayImageTk)
    displayResidualImage.image = displayImageTk

def updateResult():
    global displayResultImage
    global displayResultImageLabel
    global resultNumpy

    displayResultImage = Image.fromarray(resultNumpy)
    displayImageTk = utils.getTkinterImage(displayResultImage)
    displayResultImageLabel.config(image = displayImageTk)
    displayResultImageLabel.image = displayImageTk

def saveImageCommand(image, name):
    pilResult = Image.fromarray(image)
    pilResult.save(name)

def resetLayerColorsCommand(layers):
    for layer in layers:
        layer.resetColor()

def performShadingDecompositionCommand():
    print("performing shading decomposition")
    global shadingDecomp
    global shadingNumpy
    shadingDecomp = performDecomposition(albedoLayers, shadingNumpy)

def performResidualDecompositionCommand():
    print("performing residual decomposition")
    global residualDecomp
    global residualNumpy
    residualDecomp = performDecomposition(albedoLayers, residualNumpy)

def performDecomposition(albedoLayers, shadingNumpy):
    global participatingLayers
    print("start")
    colorList = []
    participatingLayers = []

    for layer in albedoLayers:
        color = layer.color.tolist()
        if(utils.isColorful(color)):
            colorList.append(layer.color.tolist())
            participatingLayers.append(layer)

    shadingNumpyRGB = shadingNumpy.copy()
    if (shadingNumpyRGB.shape[2] == 4):
        shadingNumpyRGB = shadingNumpyRGB[...,:3]
    decomp = solveShadingRGB(shadingNumpyRGB, colorList)
    print("end")
    return decomp

# update this to be on a segment by segment basis
def applySolvedShading(decomposition, targetLayers):
    # assumes shadingDecomp has been set properly
    # assumes participating layers are already set
    # selected shading layers probably does not need to be a global
    global shadingDecomp
    global albedoLayers
    global shadingLayers
    global shadingNumpy 

    colorList = []
    for layer in participatingLayers:
        colorList.append(layer.color.tolist())

    editedSegments = []
    nonEditedSegments = []
    for layer in targetLayers:
        if(layer.selected.get()):
            layer.image = editSingleShadingRGBSegment(decomposition, colorList, layer.image)
            editedSegments.append(layer.image)
            print("layer was selected")
        else:
            nonEditedSegments.append(layer.image)
            print("layer was not selected")
    
    allSegments = editedSegments + nonEditedSegments

    numpyImage = combineSegments(allSegments)

    return numpyImage

def applyDecompositionToShading():
    global shadingNumpy
    global shadingLayers

    newImage = applySolvedShading(shadingDecomp, shadingLayers)
    shadingNumpy = newImage.copy()

    updateShading()
    updateSelectedShading(shadingLayers)

def applyDecompositionToResidual():
    global residualNumpy
    global residualLayers

    newImage = applySolvedShading(residualDecomp, residualLayers)
    residualNumpy = newImage.copy()

    updateResidual()
    updateSelectedResidual(residualLayers)

def reconstructImage():
    global resultNumpy

    mask = np.ones_like(albedoNumpy)

    scaledAlbedo = albedoNumpy.astype(np.float64) / 255.0
    scaledShading = shadingNumpy.astype(np.float64) / 255.0
    scaledResidual = residualNumpy.astype(np.float64) / 255.0

    reconstruction = perform_recolor(mask, scaledAlbedo, scaledShading, scaledResidual, gammaSlider.get())
    reconstruction = (reconstruction * 255).astype(np.uint8)
    return reconstruction

def onGammaSliderMove(value):
    global resultNumpy
    reconstructImage()

    resultNumpy = reconstructImage()
    updateResult()


gammaSlider = tk.Scale(
    master = resultFrame,
    from_=1, 
    to=2.2,
    orient="horizontal",
    resolution= 0.1,
    label="Set Gamma",
    command = onGammaSliderMove
)

gammaSlider.set(2.2)
gammaSlider.pack()

addAlbedoLayerButton = tk.Button(
    text = "Add Albedo Color Layer",
    command=addAlbedoLayer,
    master = albedoFrame,
)

addAlbedoLayerButton.pack(side="bottom")

# should this be on a segment basis?
addShadingButton = tk.Button(
    text = "Add Shading",
    master = shadingFrame,
    command=addShadingLayer
)
addShadingButton.pack(side="bottom")

# should this be on a segment basis?
addResidualButton = tk.Button(
    text = "Add Residual",
    master = residualFrame,
    command=addResidualLayer
)
addResidualButton.pack(side="bottom")

performShadingDecompositionButton = tk.Button(
    text = "Perform Shading Decomposition",
    command=performShadingDecompositionCommand,
    master = resultFrame
)

performResidualDecompositionButton = tk.Button(
    text = "Perform Residual Decomposition",
    command=performResidualDecompositionCommand,
    master = resultFrame
)

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

performShadingDecompositionButton.pack(side="bottom")
performResidualDecompositionButton.pack(side="bottom")
saveResultImageButton.pack(side="bottom")
saveResidualImageButton.pack(side="bottom")
saveShadingImageButton.pack(side="bottom")
saveAlbedoImageButton.pack(side="bottom")

rightFrame.pack(side="right")
layerColorButtonFrame.pack()


window.mainloop()