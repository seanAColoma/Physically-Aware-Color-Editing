import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os


def rgb_to_yuv(img):
    R, G, B = img[...,0], img[...,1], img[...,2]
    Y = 0.299*R + 0.587*G + 0.114*B
    U = -0.14713*R - 0.28886*G + 0.436*B
    V = 0.615*R - 0.51499*G - 0.10001*B
    return torch.stack([Y, U, V], dim=-1)

def yuv_to_rgb(img):
    Y, U, V = img[...,0], img[...,1], img[...,2]
    R = Y + 1.13983*V
    G = Y - 0.39465*U - 0.58060*V
    B = Y + 2.03211*U
    return torch.stack([R, G, B], dim=-1)


def total_variation(T):
    dx = T[:,:,1:] - T[:,:,:-1]
    dy = T[:,1:,:] - T[:,:-1,:]
    return torch.sum(torch.sqrt(dx**2 + 1e-6)) + torch.sum(torch.sqrt(dy**2 + 1e-6))


def decompose_shading(shading, colors, n_iter=1000, lambda_nn=1e-1, lambda_sm=1e-6, lambda_sp=1e-3):
    device = shading.device
    h, w, _ = shading.shape
    n = colors.shape[0]

    # RGB → YUV
    shading_yuv = rgb_to_yuv(shading)
    colors_yuv = rgb_to_yuv(colors)

    shading_uv = shading_yuv[..., 1:]   # (H, W, 2)
    colors_uv = colors_yuv[:, 1:]       # (N, 2)

    T = torch.rand(n, h, w, device=device, requires_grad=True)
    T.data *= 0.01

    optimizer = torch.optim.Adam([T], lr=1e-2)

    for _ in range(n_iter):
        optimizer.zero_grad()

        pred_uv = torch.zeros(h, w, 2, device=device)
        for k in range(n):
            pred_uv += T[k].unsqueeze(-1) * colors_uv[k]

        E_df = torch.sum((pred_uv - shading_uv) ** 2)
        E_nn = torch.sum(F.relu(-T))
        E_sm = total_variation(T)
        E_sp = torch.sum(torch.abs(T))

        loss = E_df + lambda_nn*E_nn + lambda_sm*E_sm + lambda_sp*E_sp
        loss.backward()
        optimizer.step()

    return T.detach(), shading_yuv[..., 0]  # return Y separately


def reconstruct_shading(T, colors, Y):
    n, h, w = T.shape

    colors_yuv = rgb_to_yuv(colors)
    colors_uv = colors_yuv[:, 1:]

    uv = torch.zeros(h, w, 2)
    for k in range(n):
        uv += T[k].unsqueeze(-1) * colors_uv[k]

    yuv = torch.cat([Y.unsqueeze(-1), uv], dim=-1)
    return yuv_to_rgb(yuv)



def solveShadingYUV(input_img_np, colors_list):
    """
    Args:
        input_img_np: numpy (H, W, 3), uint8 or [0,255]
        colors_list: list of [R,G,B] (0-255)

    Returns:
        T: (N, H, W)
        Y: (H, W)
    """

    shading = torch.tensor(input_img_np / 255.0, dtype=torch.float32)

    colors = colors_list
    if [255, 255, 255] not in colors_list:
        colors = colors_list + [[255, 255, 255]]

    colors = torch.tensor(colors, dtype=torch.float32) / 255.0

    T, Y = decompose_shading(shading, colors)
    return T, Y


def editShadingYUV(T, Y, colors_edited_list):
    """
    Args:
        T: (N, H, W)
        Y: (H, W)
        colors_edited_list: list of [R,G,B]

    Returns:
        edited_img_np: (H, W, 3) uint8
    """

    colors_edited = colors_edited_list
    if T.shape[0] > len(colors_edited_list):
        colors_edited = colors_edited_list + [[255, 255, 255]]

    colors_edited = torch.tensor(colors_edited, dtype=torch.float32) / 255.0

    shading_edited = reconstruct_shading(T, colors_edited, Y)

    edited_img_np = shading_edited.clamp(0, 1).numpy()
    edited_img_np = (edited_img_np * 255).astype(np.uint8)

    return edited_img_np



if __name__ == "__main__":

    load_dir = "data/colorful_clothes_diffuse/"
    save_dir = "test/refactor/"
    os.makedirs(save_dir, exist_ok=True)

    img = Image.open(load_dir + "input_image.png").convert("RGB")
    img = np.asarray(img)

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

    T, Y = solveShadingYUV(img, colors)
    shading_edited = editShadingYUV(T, Y, colors_edited)

    Image.fromarray(shading_edited).save(save_dir + "yuv.png")

    print("Done")