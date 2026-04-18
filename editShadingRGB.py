import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os

def total_variation(T):
    dx = T[:,:,1:] - T[:,:,:-1]
    dy = T[:,1:,:] - T[:,:-1,:]
    return torch.sum(torch.sqrt(dx**2 + 1e-6)) + torch.sum(torch.sqrt(dy**2 + 1e-6))

# n_iter=1000, lambda_nn=1e3, lambda_sm=1e-4, lambda_sp=1e-10
def decompose_shading(shading, colors, n_iter=1000, lambda_nn=1e-1, lambda_sm=1e-6, lambda_sp=1e-3):
    device = shading.device
    h, w, _ = shading.shape
    n = colors.shape[0]
    T = torch.rand(n, h, w, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([T], lr=1e-2)

    for _ in range(n_iter):
        optimizer.zero_grad()

        shading_predicted = torch.zeros(h, w, 3, device=device)
        for k in range(n):
            shading_predicted += T[k].unsqueeze(-1) * colors[k]

        E_df = torch.sum((shading_predicted - shading) ** 2)
        E_nn = torch.sum(F.relu(-T))
        E_sm = total_variation(T)
        E_sp = torch.sum(torch.abs(T))

        loss = E_df + lambda_nn*E_nn + lambda_sm*E_sm + lambda_sp*E_sp
        loss.backward()
        optimizer.step()
    
    return T.detach()

def reconstruct_shading(T, colors):
    n, h, w = T.shape
    shading = torch.zeros(h, w, 3)
    for i in range(n):
        shading += T[i].unsqueeze(-1) * colors[i]
    return shading

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img) / 255.0
    return torch.tensor(img, dtype=torch.float32)

def save_image(img, path):
    img = img.clamp(0,1).numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def solveShadingRGB(input_img_np, colors_list):
    """
    Args:
        input_img_np: numpy array (H, W, 3), uint8 or float in [0,255]
        colors_list: list of [R,G,B] values (0-255)

    Returns:
        T: linear combination tensor (Number of palette colors, H, W)
    """

    # convert input image to torch
    shading = torch.tensor(input_img_np / 255.0, dtype=torch.float32)

    # convert palettes
    colors = colors_list
    if [255, 255, 255] not in colors_list:
        colors = colors_list + [[255, 255, 255]]
    colors = torch.tensor(colors, dtype=torch.float32) / 255.0
    
    # solve decomposition
    T = decompose_shading(shading, colors)
    return T


def editShadingRGB(T, colors_edited_list):
    """
    Args:
        T: linear combination tensor (Number of palette colors, H, W)
        colors_edited_list: same shape as colors_list

    Returns:
        edited_img_np: numpy array (H, W, 3), uint8 in [0,255]
    """

    # convert palettes
    colors_edited = colors_edited_list
    if T.shape[0] > len(colors_edited_list):
        colors_edited = colors_edited_list + [[255, 255, 255]]
    colors_edited = torch.tensor(colors_edited, dtype=torch.float32) / 255.0

    # reconstruct edited image
    shading_edited = reconstruct_shading(T, colors_edited)

    # convert back to numpy uint8
    edited_img_np = shading_edited.clamp(0, 1).numpy()
    edited_img_np = (edited_img_np * 255).astype(np.uint8)

    return edited_img_np

if __name__ == "__main__":

    load_dir = "data/colorful_clothes_diffuse/"
    save_dir = "test/refactor/"
    prefix = "colorful_clothes"
    os.makedirs(save_dir, exist_ok=True)

    shading = Image.open(load_dir + "input_image.png").convert('RGB')
    shading = np.asarray(shading)


    colors = [
        [137, 127, 113],
        [149, 81, 55],
        [92, 88, 98],
        [14, 2, 2],
        [159, 9, 12],
        [168, 87, 9],
        [67, 121, 32],
        [185, 113, 2],
        [208, 35, 15],
    ]

    colors_edited = [
        [137, 127, 113],
        [149, 81, 55],
        [92, 88, 98],
        [14, 2, 2],
        [159, 9, 12],
        [168, 87, 9],
        [245, 226, 20],
        [185, 113, 2],
        [208, 35, 15],
    ]

    print("Editing shading ...")

    T = solveShadingRGB(shading, colors)
    shading_edited = editShadingRGB(T, colors_edited)
    Image.fromarray(shading_edited).save(save_dir+"rgb.png")

    print("Done")
