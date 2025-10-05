from utils import zero_out, apply_flux_guidance, load_torch_file
import torch
from apply_clip import run_clip
from apply_style import load_style_model, apply_stylemodel ,STYLE_MODEL_PATH, CLIPOutputWrapper, ReduxImageEncoder, StyleModel
import joblib
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL, FluxTransformer2DModel
import math
from tqdm import tqdm
from safetensors.torch import load_file
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

def load_flux_model(ckpt_path, device="cuda"):
    print("Loading FLUX model from safetensors (meta-safe)...")

    config = {
        "patch_size": 1,
        "in_channels": 64,
        "num_layers": 19,
        "num_single_layers": 38,
        "attention_head_dim": 128,
        "num_attention_heads": 24,
        "joint_attention_dim": 4096,
        "pooled_projection_dim": 768,
        "guidance_embeds": False,
        "axes_dims_rope": (16, 56, 56)
    }

    with init_empty_weights():
        model = FluxTransformer2DModel(**config)

    model = model.to_empty(device=device) 
    model = model.to(dtype=torch.float16)  
    state_dict = load_file(ckpt_path, device="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    torch.cuda.empty_cache()
    print("✅ FLUX model loaded successfully (no memory spike, meta-safe).")

    return model

def kl_optimal_scheduler(n: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    adj_idxs = torch.arange(n, dtype=torch.float).div_(n - 1)
    sigmas = adj_idxs.new_zeros(n + 1)
    sigmas[:-1] = (adj_idxs * math.atan(sigma_min) + (1 - adj_idxs) * math.atan(sigma_max)).tan_()
    return sigmas

def denoise_with_cfg(model, x, sigma, positive, negative, cfg):
    latent_model_input = torch.cat([x] * 2)
    text_embeddings = torch.cat([positive[0][0], negative[0][0]])
    
    timestep = torch.full((latent_model_input.shape[0],), sigma.item(), device=x.device, dtype=x.dtype)

    pooled_prompt_embeds = torch.cat([positive[0][1].get("pooled_output"), negative[0][1].get("pooled_output")])
    latent_image = torch.cat([positive[0][1].get("concat_latent_image"), negative[0][1].get("concat_latent_image")])

    noise_pred = model(
        latent_model_input, 
        timestep=timestep, 
        encoder_hidden_states=text_embeddings,
        pooled_projections=pooled_prompt_embeds,
        latent_image=latent_image
    ).sample
    
    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
    
    cfg_noise_pred = noise_pred_uncond + (noise_pred_cond - noise_pred_uncond) * cfg
    
    denoised = x - cfg_noise_pred * sigma
    
    return denoised

def euler_sampler_loop(model, noise, sigmas, positive, negative, cfg, callback=None):
    x = noise * sigmas[0]
    for i in tqdm(range(len(sigmas) - 1)):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i+1]
        
        denoised = denoise_with_cfg(model, x, sigma_cur, positive, negative, cfg)
        
        derivative = (x - denoised) / sigma_cur
        dt = sigma_next - sigma_cur
        x = x + derivative * dt
        
        if callback is not None:
            callback(i, x, denoised)
            
    return x

def run_k_sampler(model, seed, steps, cfg, positive, negative, latent, device):
    samples = latent['samples'].to(device) # Ensure latents are on the correct device for the model
    noise = torch.randn(samples.shape, generator=torch.manual_seed(seed), device=device, dtype=torch.float16)
    
    sigma_min = 0.002
    sigma_max = 80.0
    sigmas = kl_optimal_scheduler(steps, sigma_min, sigma_max).to(device).to(torch.float16)

    denoised_latents = euler_sampler_loop(model, noise, sigmas, positive, negative, cfg)
    
    return denoised_latents

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
    
    for i in range(3):
        pixels[:,:,:,i] -= 0.5
        pixels[:,:,:,i] *= m
        pixels[:,:,:,i] += 0.5

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
    style_model.model = style_model.model.to(device)
    
    conditioning_cpu = joblib.load("/teamspace/studios/this_studio/.porting/models/conditioning/text_cache.conditioning")
    conditioning = []
    for t in conditioning_cpu:
        cond_tensor, cond_dict = t
        new_dict = {}
        for k, v in cond_dict.items():
            if isinstance(v, torch.Tensor):
                new_dict[k] = v.to(device)
            else:
                new_dict[k] = v
        conditioning.append([cond_tensor.to(device), new_dict])

    out_cond, = apply_stylemodel(conditioning, style_model, clip_vis_output, 1, "multiply")
    
    # Clean up style and clip models after use
    del style_model, clip_vis_tensor, clip_vis_output
    torch.cuda.empty_cache()
    print("Cleaned up style and clip models from memory.")
    
    negative_conds, = zero_out(out_cond)
    positive_conds, = apply_flux_guidance(out_cond, 40.3)

    pixels_pil = Image.open("/teamspace/studios/this_studio/.porting/imgs/img_pixels_concat.jpg").convert("RGB")
    pixels_np = np.array(pixels_pil).astype(np.float32) / 255.0
    pixels = torch.from_numpy(pixels_np).unsqueeze(0).to(torch.float16).to(device)

    mask_pil = Image.open("/teamspace/studios/this_studio/.porting/imgs/img_mask_concat.jpg").convert("L")
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask_np).unsqueeze(0).to(torch.float16).to(device)
    
    vae_path = "/teamspace/studios/this_studio/.porting/models/vae/ae.safetensors"
    vae = load_vae_model(vae_path, device)

    new_positive_conds, new_negative_conds, latents = inpaint(positive_conds, negative_conds, pixels, vae, mask)
    
    # Offload VAE to CPU to free up VRAM for the large FLUX model
    vae.to("cpu")
    del pixels, mask
    torch.cuda.empty_cache()
    print("Offloaded VAE to CPU.")

    flux_model_path = "/teamspace/studios/this_studio/.porting/models/unet/flux1-fill-dev-fp8.safetensors"
    flux_model = load_flux_model(flux_model_path, device)

    final_latents = run_k_sampler(flux_model, seed=42, steps=25, cfg=1.8, positive=new_positive_conds, negative=new_negative_conds, latent=latents, device=device)

    # Clean up the main FLUX model
    del flux_model
    torch.cuda.empty_cache()
    print("Cleaned up FLUX model from memory.")

    # Move VAE back to GPU for decoding
    vae.to(device)
    print("Moved VAE back to GPU for final decoding.")

    final_latents = final_latents.to(device) / vae.config.scaling_factor
    decoded_image = vae.decode(final_latents).sample
    decoded_image = (decoded_image.clamp(-1, 1) + 1) / 2
    decoded_image = (decoded_image * 255).cpu().numpy().astype(np.uint8)
    
    image = Image.fromarray(decoded_image[0].transpose(1, 2, 0))
    image.save("final_output.png")

    print("Successfully generated final image: final_output.png")

