import numpy as np
from PIL import Image

def reconstructFromLayers(alphaLayers):
    pilImages = []
    for layer in alphaLayers:
        pilImages.append(Image.fromarray(layer.image))

    result = pilImages[0]
    for pilLayer in pilImages[1:]:
        result = Image.alpha_composite(result, pilLayer)
    return np.asarray(result)