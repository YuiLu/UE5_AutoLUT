# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************

# Adapted from https://github.com/guoyww/AnimateDiff
import os
import imageio
import numpy as np

import torch
import torchvision

from PIL import Image, ImageFilter
from typing import Union
from tqdm import tqdm
from einops import rearrange
import torch.distributed as dist


def save_cube_lut(lut: np.ndarray, path: str, title: str = "AutoLUT", domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0), lut_size: int = 16):
    """Save a 3D LUT to a .cube file.

    Accepts HWC 2D layout used by PIL's Color3DLUT (e.g. 64x64x3 for lut_size=16).
    
    PIL's Color3DLUT uses a specific memory layout where the table is indexed as:
        table[r + g*size + b*size*size] -> (R_out, G_out, B_out)
    i.e., R changes fastest, then G, then B.
    
    .cube file format requires the opposite ordering:
        R changes slowest, G in the middle, B changes fastest.
    
    This function converts from PIL's layout to .cube standard layout.
    """
    lut_arr = np.asarray(lut, dtype=np.float32)
    
    if lut_arr.ndim == 3 and lut_arr.shape[-1] == 3:
        # Input shape: (size*size, size, 3) flattened to (size^3, 3)
        # PIL layout: index = r + g*size + b*size^2
        expected = lut_size ** 3
        lut_flat = lut_arr.reshape(-1, 3)
        if lut_flat.shape[0] != expected:
            raise ValueError(f"Expected LUT with {expected} entries, got {lut_flat.shape[0]}")
        
        # Reshape to (B, G, R, 3) indexing based on PIL's layout
        # PIL: flat_index = r + g*size + b*size^2
        # So reshape to (size, size, size, 3) gives us [b, g, r, :]
        lut_bgr = lut_flat.reshape(lut_size, lut_size, lut_size, 3)
        
        # .cube needs (R, G, B, 3) with R slowest, B fastest
        # Transpose from [b, g, r, :] to [r, g, b, :]
        lut_cube = np.transpose(lut_bgr, (2, 1, 0, 3))
        lut_rgb = lut_cube.reshape(-1, 3)
    elif lut_arr.ndim == 1:
        if lut_arr.size != 3 * (lut_size ** 3):
            raise ValueError(f"Expected flat LUT of length {3 * (lut_size ** 3)}, got {lut_arr.size}")
        # Same conversion for flat input
        lut_bgr = lut_arr.reshape(lut_size, lut_size, lut_size, 3)
        lut_cube = np.transpose(lut_bgr, (2, 1, 0, 3))
        lut_rgb = lut_cube.reshape(-1, 3)
    else:
        raise ValueError(f"Unsupported LUT shape: {lut_arr.shape}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if title:
            f.write(f'TITLE "{title}"\n')
        f.write(f"LUT_3D_SIZE {lut_size}\n")
        f.write(f"DOMAIN_MIN {domain_min[0]} {domain_min[1]} {domain_min[2]}\n")
        f.write(f"DOMAIN_MAX {domain_max[0]} {domain_max[1]} {domain_max[2]}\n")
        for r, g, b in lut_rgb:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=25):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def save_images_grid(images: torch.Tensor, path: str):
    assert images.shape[2] == 1 # no time dimension
    images = images.squeeze(2)
    grid = torchvision.utils.make_grid(images)
    grid = (grid * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(grid).save(path)

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


def video2images(path, step=4, length=16, start=0):
    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        frames.append(np.array(frame))
    frames = frames[start::step][:length]
    return frames


def images2video(video, path, fps=8):
    imageio.mimsave(path, video, fps=fps)
    return


tensor_interpolation = None

def get_tensor_interpolation_method():
    return tensor_interpolation

def set_tensor_interpolation_method(is_slerp):
    global tensor_interpolation
    tensor_interpolation = slerp if is_slerp else linear

def linear(v1, v2, t):
    return (1.0 - t) * v1 + t * v2

def slerp(
    v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995
) -> torch.Tensor:
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        #logger.info(f'warning: v0 and v1 close to parallel, using linear interpolation instead.')
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()


def vars(src, ref):

    r, z = src.reshape([-1, src.shape[-1]]).T, ref.reshape([-1, ref.shape[2]]).T

    cov_r, cov_z = np.cov(r), np.cov(z)

    mu_r, mu_z = r.mean(axis=1)[..., np.newaxis], z.mean(axis=1)[..., np.newaxis]
    eig_val_r, eig_vec_r = np.linalg.eig(cov_r)
    eig_val_r[eig_val_r < 0] = 0
    val_r = np.diag(np.sqrt(eig_val_r[::-1]))
    vec_r = np.array(eig_vec_r[:, ::-1])
    inv_r = np.diag(1. / (np.diag(val_r + np.spacing(1))))

    mat_c = val_r @ vec_r.T @ cov_z @ vec_r @ val_r
    eig_val_c, eig_vec_c = np.linalg.eig(mat_c)
    eig_val_c[eig_val_c < 0] = 0
    val_c = np.diag(np.sqrt(eig_val_c))

    transfer_mat = vec_r @ inv_r @ eig_vec_c @ val_c @ eig_vec_c.T @ inv_r @ vec_r.T
    return [cov_r, cov_z, mu_r, mu_z, transfer_mat]

def transfer(src, ref, variables):
    cov_r, cov_z, mu_r, mu_z, transfer_mat = variables
    r, z = src.reshape([-1, src.shape[2]]).T, ref.reshape([-1, ref.shape[2]]).T

    res = np.dot(transfer_mat, r - mu_r) + mu_z

    res = res.T.reshape(src.shape)

    return res

def preprocess(src, ref, size, ncc):
    """
    Preprocess input video with optional color transfer.
    
    Returns:
        input_video: preprocessed video frames (uint8)
        input_video_resize: resized preprocessed frames for model input
        preprocess_params: dict with color transfer parameters (for LUT generation)
                          None if ncc=True (no color correction)
    """
    input_video = [np.array(Image.fromarray(c)) for c in src][:480]
    preprocess_params = None
    
    if not ncc:
        output_frames = []
        input_video_cc = [np.array(Image.fromarray(c).resize((256, 256))) for c in input_video]
        variables = vars(np.array(input_video_cc), ref)
        for i, frame in enumerate(input_video):
            img_res = transfer(frame, ref, variables)
            output_frames.append(img_res)
        output_frames = np.array(output_frames) 
        
        # Record normalization parameters for LUT generation
        out_min, out_max = output_frames.min(), output_frames.max()
        output_frames = (output_frames - out_min) / (out_max - out_min)
        input_video = (output_frames * 255.).astype(np.uint8)
        
        # Store parameters for combined LUT generation
        preprocess_params = {
            'transfer_mat': variables[4],  # transfer_mat
            'mu_r': variables[2],          # source mean
            'mu_z': variables[3],          # reference mean
            'out_min': out_min,
            'out_max': out_max,
        }

    input_video_resize = [np.array(Image.fromarray(c).resize((size, size))) for c in input_video]

    return input_video, input_video_resize, preprocess_params


def apply_preprocess_to_color(rgb_float, preprocess_params):
    """
    Apply preprocess color transform to a single RGB value (0-1 range).
    Used for generating combined LUT.
    
    Args:
        rgb_float: numpy array of shape (3,) with values in [0, 1]
        preprocess_params: dict from preprocess() function
    
    Returns:
        transformed RGB value in [0, 1] range
    """
    if preprocess_params is None:
        return rgb_float
    
    transfer_mat = preprocess_params['transfer_mat']
    mu_r = preprocess_params['mu_r']
    mu_z = preprocess_params['mu_z']
    out_min = preprocess_params['out_min']
    out_max = preprocess_params['out_max']
    
    # Convert to 0-255 range for transform (matching original preprocess)
    rgb_255 = rgb_float * 255.0
    
    # Apply color transfer: res = transfer_mat @ (pixel - mu_r) + mu_z
    pixel = rgb_255.reshape(3, 1)
    transformed = np.dot(transfer_mat, pixel - mu_r) + mu_z
    transformed = transformed.flatten()
    
    # Apply same normalization as preprocess
    normalized = (transformed - out_min) / (out_max - out_min)
    
    return np.clip(normalized, 0.0, 1.0)


def generate_combined_lut(grading_lut_hwc, preprocess_params, lut_size=16):
    """
    Generate a combined LUT that includes both preprocess color transfer and grading.
    
    This LUT can be used in external software to get the same result as the output video.
    
    Args:
        grading_lut_hwc: the grading LUT in HWC format (size*size, size, 3) = (64, 64, 3) for size=16
                         This is the format used by PIL's identity_table
        preprocess_params: parameters from preprocess() function
        lut_size: LUT size (default 16)
    
    Returns:
        combined_lut_hwc: combined LUT in same format as input
    """
    if preprocess_params is None:
        # No preprocess was applied, return original LUT
        return grading_lut_hwc
    
    # Create PIL filter for grading LUT
    grading_filter = ImageFilter.Color3DLUT(lut_size, grading_lut_hwc.flatten().tolist())
    
    # Generate combined LUT
    # The HWC format is reshaped from (4096*3,) to (64, 64, 3)
    # PIL flat index = r + g*16 + b*256
    # HWC index: h = flat_idx // 64, w = flat_idx % 64
    combined_lut = np.zeros_like(grading_lut_hwc)
    
    for b in range(lut_size):
        for g in range(lut_size):
            for r in range(lut_size):
                # Input color (normalized to 0-1)
                input_rgb = np.array([r, g, b], dtype=np.float32) / (lut_size - 1)
                
                # Step 1: Apply preprocess color transform
                preprocessed_rgb = apply_preprocess_to_color(input_rgb, preprocess_params)
                
                # Step 2: Apply grading LUT
                # Convert to uint8 image for PIL filter
                preprocessed_uint8 = (preprocessed_rgb * 255).astype(np.uint8)
                img = Image.fromarray(preprocessed_uint8.reshape(1, 1, 3))
                graded_img = img.filter(grading_filter)
                graded_rgb = np.array(graded_img).flatten() / 255.0
                
                # Store in combined LUT
                # PIL flat_idx = r + g*16 + b*256
                flat_idx = r + g * lut_size + b * lut_size * lut_size
                # HWC reshape (4096,3) -> (64, 64, 3): row-major order
                h = flat_idx // 64
                w = flat_idx % 64
                combined_lut[h, w] = graded_rgb
    
    return combined_lut
