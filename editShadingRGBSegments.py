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
    pil_img = Image.open(path).convert('RGBA')
    img = np.array(pil_img) / 255.0
    img = torch.tensor(img, dtype=torch.float32)

    rgb = img[..., :3]
    alpha = img[..., 3:]
    return rgb, alpha, pil_img

def tensor_to_PIL(rgb, alpha):
    img = torch.cat([rgb, alpha], dim=-1)
    img = img.clamp(0,1).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img, mode='RGBA')


def process_segments_rgb(segment_paths, edit_set, T, colors_edited, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    final_img = None  # PIL image

    full_rgb = reconstruct_shading(T, colors_edited)
    for path in segment_paths:
        fname = os.path.basename(path)

        seg_rgb, seg_alpha, seg_pil = load_image(path)
        mask = seg_alpha > 1e-3

        if fname in edit_set:
            seg_rgb[mask.squeeze(-1) > 0] = full_rgb[mask.squeeze(-1) > 0]
            seg_pil = tensor_to_PIL(seg_rgb, seg_alpha)
            out_name = fname.replace(".png", "_edited.png")

            n = T.shape[0]
            for k in range(n):
                component = T[k].unsqueeze(-1) * colors_edited[k]  # (H,W,3)
                comp_rgb = torch.zeros_like(component)
                comp_rgb[mask.squeeze(-1) > 0] = component[mask.squeeze(-1) > 0]
                comp_alpha = torch.zeros_like(seg_alpha)
                comp_alpha[mask] = seg_alpha[mask]
                comp_pil = tensor_to_PIL(comp_rgb, comp_alpha)
                comp_name = fname.replace(".png", f"_component_{k}.png")
                comp_pil.save(os.path.join(save_dir, comp_name))

        else:
            out_name = fname
        seg_pil.save(os.path.join(save_dir, out_name))


        # Composite using PIL
        if final_img is None:
            final_img = seg_pil
        else:
            final_img = Image.alpha_composite(final_img, seg_pil)

    return final_img



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

def editShadingRGBSegments(T, colors_edited_list, segments_rgba_np):
    """
    Args:
        T: (n, H, W)
        colors_edited_list: list of [R,G,B]
        segments_rgba_np: list of numpy arrays (H, W, 4)

    Returns:
        edited_segments: list of numpy arrays (H, W, 4)
    """

    # convert palettes
    if T.shape[0] > len(colors_edited_list):
        colors_edited_list = colors_edited_list + [[255, 255, 255]]
    colors_edited = torch.tensor(colors_edited_list, dtype=torch.float32) / 255.0

    full_rgb = reconstruct_shading(T, colors_edited)
    edited_segments = []

    for seg_np in segments_rgba_np:

        seg = torch.tensor(seg_np / 255.0, dtype=torch.float32)
        seg_rgb = seg[..., :3]
        seg_alpha = seg[..., 3:]

        mask = seg_alpha > 1e-3
        seg_rgb[mask.squeeze(-1)] = full_rgb[mask.squeeze(-1)]
        seg_rgba = torch.cat([seg_rgb, seg_alpha], dim=-1)

        seg_np_out = seg_rgba.clamp(0, 1).numpy()
        seg_np_out = (seg_np_out * 255).astype(np.uint8)
        edited_segments.append(seg_np_out)

    return edited_segments

def combineSegments(segs):
    """
    Args:
        segs: list of (H, W, 4) RGBA np arrays

    Returns:
        final_image: (H, W, 4) RGBA np array image
    """
    h, w, _ = segs[0].shape
    final_rgb = np.zeros((h, w, 3), dtype=np.float32)
    final_alpha = np.zeros((h, w, 1), dtype=np.float32)

    for seg in segs:
        seg = seg.astype(np.float32) / 255.0
        seg_rgb, seg_alpha = seg[..., :3], seg[..., 3:]
        final_rgb += seg_rgb * seg_alpha
        final_alpha += seg_alpha

    final_rgb = final_rgb / (final_alpha + 1e-6)
    final_alpha = np.clip(final_alpha, 0.0, 1.0)
    final_image = np.concatenate([final_rgb, final_alpha], axis=-1)
    final_image = (final_image * 255).astype(np.uint8)
    return final_image


if __name__ == "__main__":

    # CONFIGS

    load_dir = "data/colorful_clothes_diffuse/"
    save_dir = "test/refactor/rgb_seg/"
    prefix = "colorful_clothes_diffuse"
    os.makedirs(save_dir, exist_ok=True)

    edit_set = [
        "FinalLayers_03.png",
        "FinalLayers_05.png",
        "FinalLayers_06.png",
    ]
    edit_set = [load_dir + "output_layers/" + fname for fname in edit_set]

    non_edit_set = [
        "FinalLayers_00.png",
        "FinalLayers_01.png",
        "FinalLayers_02.png",
        "FinalLayers_04.png",
        "FinalLayers_07.png",
    ]
    non_edit_set = [load_dir + "output_layers/" + fname for fname in non_edit_set]

    shading = Image.open(load_dir + "input_image.png").convert('RGB')
    shading = np.asarray(shading)

    seg_list = [np.array(Image.open(seg)) for seg in edit_set]
    seg_list_unedited = [np.array(Image.open(seg)) for seg in non_edit_set]


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
    shading_edited = editShadingRGBSegments(T, colors_edited, seg_list)
    for i, seg in enumerate(shading_edited):
        Image.fromarray(seg).save(save_dir + f"layer{i}.png")

    img = combineSegments(seg_list_unedited + shading_edited)
    Image.fromarray(img).save(save_dir + "shading.png")
    print("Done")


