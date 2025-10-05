from utils import zero_out, apply_flux_guidance, load_torch_file
import torch
from apply_clip import run_clip
from apply_style import load_style_model, apply_stylemodel ,STYLE_MODEL_PATH, CLIPOutputWrapper
import joblib
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL

# Conditioning is 40.3

def conditioning_set_values(conditioning, values={}, append=False):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            val = values[k]
            if append:
                old_val = n[1].get(k, None)
                if old_val is not None:
                    val = old_val + val

            n[1][k] = val
        c.append(n)

    return c

def load_vae_model(ckpt_path, device):
    """
    Loads the VAE model from a single safetensors file using the diffusers library
    and moves it to the specified device.
    """
    print("Loading VAE model...")
    vae = AutoencoderKL.from_single_file(
        ckpt_path,
        torch_dtype=torch.float16,
        latent_channels=16,
        low_cpu_mem_usage=False,
    )
    vae = vae.to(device)
    print("VAE model loaded successfully.")
    return vae

def inpaint(positive, negative, pixels, vae, mask, noise_mask=True):
    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")
    orig_pixels = pixels
    pixels = orig_pixels.clone()
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
        mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]
    
    m = (1.0 - mask.round()).squeeze(1)
    
    # This loop assumes a (Batch, Height, Width, Channel) format
    for i in range(3):
        pixels[:,:,:,i] -= 0.5
        pixels[:,:,:,i] *= m
        pixels[:,:,:,i] += 0.5

    # The diffusers VAE expects (Batch, Channel, Height, Width), so we permute the tensor dimensions.
    # It also returns a distribution object, so we sample from it to get the latent tensor.
    concat_latent = vae.encode(pixels.permute(0, 3, 1, 2)).latent_dist.sample()
    orig_latent = vae.encode(orig_pixels.permute(0, 3, 1, 2)).latent_dist.sample()

    out_latent = {}
    out_latent["samples"] = orig_latent
    if noise_mask:
        out_latent["noise_mask"] = mask
    out = []
    for conditioning in [positive, negative]:
        c = conditioning_set_values(conditioning, {"concat_latent_image": concat_latent,
                                                     "concat_mask": mask})
        out.append(c)
    return (out[0], out[1], out_latent)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    clip_vis_tensor = run_clip().to(device)
    clip_vis_output = CLIPOutputWrapper(clip_vis_tensor)
    
    style_model = load_style_model(STYLE_MODEL_PATH)
    # Move the style model's internal nn.Module to the correct device
    style_model.model = style_model.model.to(device)
    
    conditioning_cpu = joblib.load("/teamspace/studios/this_studio/.porting/models/conditioning/text_cache.conditioning")
    # Move the tensors within the conditioning list to the correct device
    conditioning = [[t[0].to(device), t[1]] for t in conditioning_cpu]


    out_cond, = apply_stylemodel(conditioning, style_model, clip_vis_output, 1, "multiply")
    
    negative_conds, = zero_out(out_cond)
    positive_conds, = apply_flux_guidance(out_cond, 40.3)

    pixels_pil = Image.open("/teamspace/studios/this_studio/.porting/imgs/img_pixels_concat.jpg").convert("RGB")
    pixels_np = np.array(pixels_pil).astype(np.float32) / 255.0
    # Move tensor to the selected device (GPU or CPU)
    pixels = torch.from_numpy(pixels_np).unsqueeze(0).to(torch.float16).to(device)

    mask_pil = Image.open("/teamspace/studios/this_studio/.porting/imgs/img_mask_concat.jpg").convert("L")
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    # Move tensor to the selected device (GPU or CPU)
    mask = torch.from_numpy(mask_np).unsqueeze(0).to(torch.float16).to(device)
    
    vae_path = "/teamspace/studios/this_studio/.porting/models/vae/ae.safetensors"
    vae = load_vae_model(vae_path, device)

    new_positive_conds, new_negative_conds, latents = inpaint(positive_conds, negative_conds, pixels, vae, mask)

    print("Positive conditioning:",new_positive_conds)
    print("Negative conditioning:",new_negative_conds)
    print("Latents:",latents)

    print("Positive conditioning shape:",new_positive_conds.shape)
    print("Negative conditioning shape:",new_negative_conds.shape)
    print("Latents shape:",latents.shape)
    print("Successfully applied the style model.")

