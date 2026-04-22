from chrislib.general import view, tile_imgs, invert
from chrislib.data_util import load_image
from PIL import Image
import numpy as np
import torch
import os

from intrinsic.pipeline import load_models, run_pipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = load_models('v2', device=device)

def decompose(image_path):
    image = load_image(image_path)

    results = run_pipeline(models, image, device=device)

    img = results['image']
    alb = results['hr_alb']
    dif = results['dif_shd']   # raw linear shading, can be > 1
    res = results['residual']  # raw residual, can be negative

    return img, alb, dif, res

def saveDecomposedImages(img, alb, dif, res): 
    os.makedirs("output", exist_ok=True)
    dif_display = 1 - invert(dif)
    Image.fromarray((np.clip(alb, 0, 1) * 255).astype(np.uint8)).save('output/albedo.png')
    Image.fromarray((np.clip(dif_display, 0, 1) * 255).astype(np.uint8)).save('output/diffuse_shading.png')
    Image.fromarray((np.clip(res, 0, 1) * 255).astype(np.uint8)).save('output/residual.png')

if __name__ == '__main__':
    img, alb, dif, res = decompose('sample/albedo.webp')

    # save for reconstruction (preserves full HDR range)
    np.save('output/albedo.npy', alb)
    np.save('output/diffuse_shading.npy', dif)
    np.save('output/residual.npy', res)

    # save PNGs for viewing only
    dif_display = 1 - invert(dif)
    Image.fromarray((np.clip(alb, 0, 1) * 255).astype(np.uint8)).save('output/albedo.png')
    Image.fromarray((np.clip(dif_display, 0, 1) * 255).astype(np.uint8)).save('output/diffuse_shading.png')
    Image.fromarray((np.clip(res, 0, 1) * 255).astype(np.uint8)).save('output/residual.png')

