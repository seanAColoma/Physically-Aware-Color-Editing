import numpy as np
from PIL import Image
from chrislib.general import match_scale
from chrislib.data_util import load_image

def perform_recolor(msk, alb, shd, res, shd_power=1.0, recolor=None, gamma = 2.2):
    # msk - numpy array (HxWx1) denoting the region to perform the edit
    # alb - linear albedo of the image
    # shd - linear shading of the image
    # res - residual component of the image
    # shd_power - exponent to apply to the shading (<1 for more diffuse, >1 for more specular)
    # recolor - a texture to apply to the edited region, no recoloring is performed if set to None

    if recolor is None:
        our_new_alb = alb
    else:
        recolor = match_scale(recolor, alb, msk.astype(bool))
        our_new_alb = ((1.0 - msk) * alb) + (msk * recolor)

    masked_shd = msk * (shd ** shd_power)
    new_shd = ((1.0 - msk) * shd) + masked_shd

    # note: gamma changed to 2.0, reconstructions appeared too washed out with 2.2
    recolored = (our_new_alb * new_shd) ** (1/gamma)

    # reconstruct: I = (Ad * Sd)^(1/2.2) + R
    return np.clip(recolored + res, 0, 1)

if __name__ == '__main__':
    alb = np.load('output/albedo.npy')
    shd = np.load('output/diffuse_shading.npy')
    res = np.load('output/residual.npy')

    msk = np.ones_like(alb)
    tex = None

    reconstructed = perform_recolor(msk, alb, shd, res, shd_power=1.5, recolor=tex)
    result = (reconstructed * 255).astype(np.uint8)

    Image.fromarray(result).save('sample/I_reconstructed.webp')

