import numpy as np
from PIL import Image

def reconstructFromLayers(alphaLayers):
    pilImages = []

    if(len(alphaLayers) < 1):
        return np.zeros((512, 512, 3), dtype=np.uint8)

    for layer in alphaLayers:
        pilImages.append(Image.fromarray(layer))

    result = pilImages[0]
    for pilLayer in pilImages[1:]:
        result = Image.alpha_composite(result, pilLayer)
    return np.asarray(result)