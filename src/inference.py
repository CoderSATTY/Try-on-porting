import torch
import numpy as np
import os
import math
from PIL import Image
from einops import rearrange
from numpy import poly1d
from safetensors.torch import load_file
from tqdm import tqdm
from types import SimpleNamespace
from torch import nn
import copy

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else "."
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_DIR = os.path.join(BASE_DIR, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)

# --- File Paths ---
INPUT_IMAGE_PATH = os.path.join(IMG_DIR, "input_image.png")
MASK_IMAGE_PATH = os.path.join(IMG_DIR, "mask_image.png")
OUTPUT_IMAGE_PATH = os.path.join(IMG_DIR, "output_image.png")

# --- Inference Parameters ---
PROMPT = "A majestic lion, cinematic lighting"
TEA_CACHE_THRESHOLD = 0.4 
TEA_CACHE_START_PERCENT = 0.0
TEA_CACHE_END_PERCENT = 1.0

# --- Helper & Model Definition ---

def apply_mod(tensor, m_mult, m_add=None, modulation_dims=None):
    if modulation_dims is None:
        return torch.addcmul(m_add, tensor, m_mult) if m_add is not None else tensor * m_mult
    else:
        for d in modulation_dims:
            tensor[:, d[0]:d[1]] *= m_mult[:, d[2]]
            if m_add is not None:
                tensor[:, d[0]:d[1]] += m_add[:, d[2]]
        return tensor

def timestep_embedding(t: torch.Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(t) if torch.is_floating_point(t) else embedding

class FluxUNet(nn.Module):
    def __init__(self, in_channels=384, out_channels=16, model_channels=3072):
        super().__init__()
        self.params = SimpleNamespace(vec_in_dim=256, guidance_embed=False, in_channels=in_channels, out_channels=out_channels)
        self.img_in = nn.Linear(in_channels, model_channels)
        self.time_in = nn.Linear(256, model_channels * 4)
        self.vector_in = nn.Linear(256, model_channels * 4)
        self.txt_in = nn.Linear(4096, model_channels) # Corrected to T5 only
        self.pe_embedder = nn.Identity()
        
        class MockBlock(nn.Module):
            def __init__(self, model_channels):
                super().__init__()
                self.img_mod = lambda x: (SimpleNamespace(scale=torch.randn(1, model_channels * 6), shift=torch.randn(1, model_channels * 6)), None)
                self.img_norm1 = nn.Identity()
            def forward(self, img, txt, vec, pe, attn_mask): return img, txt
        
        class IdentityBlock(nn.Module):
            def forward(self, img, vec, pe, attn_mask): return img

        self.double_blocks = nn.ModuleList([MockBlock(model_channels)])
        self.single_blocks = nn.ModuleList([IdentityBlock()])
        self.final_layer = lambda x, vec: x
    
    def forward(self, *args, **kwargs):
        # The forward pass expects a tensor with the corrected in_channels dimension
        img_arg = args[0]
        if img_arg.shape[-1] != self.params.in_channels:
             # This is a mock; in a real model, we'd reshape or patch embed
             img_arg = torch.randn(img_arg.shape[0], img_arg.shape[1], self.params.in_channels, device=img_arg.device, dtype=img_arg.dtype)
        return torch.randn_like(img_arg)

# --- TeaCache Implementation (ComfyUI Style) ---

SUPPORTED_MODELS_COEFFICIENTS = {"flux": [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]}

def teacache_flux_forward(self, img: torch.Tensor, img_ids: torch.Tensor, txt: torch.Tensor, txt_ids: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor, guidance: torch.Tensor = None, control=None, transformer_options={}, attn_mask: torch.Tensor = None) -> torch.Tensor:
    opts = transformer_options
    if y is None: y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    
    # Mocking a patch embedding for the shape mismatch
    if img.shape[-1] != self.params.in_channels:
        img = torch.randn(img.shape[0], img.shape[1], self.params.in_channels, device=img.device, dtype=img.dtype)

    img_mod1, _ = self.double_blocks[0].img_mod(vec)
    modulated_inp = self.double_blocks[0].img_norm1(self.img_in(img))
    modulated_inp = apply_mod(modulated_inp, (1 + img_mod1.scale), img_mod1.shift).to(opts.get("cache_device"))

    if not hasattr(self, 'accumulated_rel_l1_distance'):
        should_calc = True
        self.accumulated_rel_l1_distance = 0
    else:
        try:
            dist = ((modulated_inp - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean())
            self.accumulated_rel_l1_distance += poly1d(opts.get("coefficients"), dist.item()).abs()
            should_calc = self.accumulated_rel_l1_distance >= opts.get("rel_l1_thresh")
            if should_calc: self.accumulated_rel_l1_distance = 0
        except:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
    
    self.previous_modulated_input = modulated_inp.clone()
    
    if opts.get("enable_teacache", True) and not should_calc:
        tqdm.write("   -> [TeaCache] SKIP: Reusing previous calculation.")
        return img + self.previous_residual.to(img.device)
    else:
        if opts.get("enable_teacache", True): tqdm.write("   -> [TeaCache] COMPUTE: Change threshold exceeded.")
        output = self.forward_orig(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids, timesteps=timesteps, y=y, guidance=guidance, control=control, transformer_options=transformer_options, attn_mask=attn_mask)
        self.previous_residual = (output - img).clone().to(opts.get("cache_device"))
        return output

class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.model_options = {}
    def clone(self): return copy.deepcopy(self)
    def get_model_object(self, name): return self.model
    def set_model_unet_function_wrapper(self, wrapper_func): self.model_unet_function = wrapper_func

class TeaCache:
    FUNCTION = "apply_teacache"
    def apply_teacache(self, model, model_type: str, rel_l1_thresh: float, start_percent: float, end_percent: float, cache_device: str):
        if rel_l1_thresh == 0: return (model,)
        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options: new_model.model_options['transformer_options'] = {}
        
        opts = new_model.model_options["transformer_options"]
        opts["rel_l1_thresh"] = rel_l1_thresh
        opts["coefficients"] = SUPPORTED_MODELS_COEFFICIENTS[model_type]
        opts["cache_device"] = torch.device(cache_device)
        
        diffusion_model = new_model.get_model_object("diffusion_model")
        diffusion_model.forward_orig = diffusion_model.forward
        diffusion_model.forward = teacache_flux_forward.__get__(diffusion_model, diffusion_model.__class__)

        def unet_wrapper_function(model_function, kwargs):
            c = kwargs["c"]
            sigmas = c["transformer_options"]["sample_sigmas"]
            current_step_index = (sigmas == kwargs["timestep"][0]).nonzero()
            current_step_index = current_step_index.item() if len(current_step_index) > 0 else 0
            
            if current_step_index == 0 and hasattr(diffusion_model, 'accumulated_rel_l1_distance'):
                delattr(diffusion_model, 'accumulated_rel_l1_distance')
            
            current_percent = current_step_index / (len(sigmas) - 1)
            c["transformer_options"]["enable_teacache"] = start_percent <= current_percent <= end_percent
            return model_function(**kwargs['c']['main_call_kwargs'])

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        return (new_model,)

# --- Main Inference Pipeline ---
def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"Using device: {device}")

    from diffusers import AutoencoderKL
    from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModelWithProjection, CLIPTokenizer, T5Config, CLIPTextConfig

    print("Loading models from local directory...")
    try:
        t5_config = T5Config.from_pretrained("google/flan-t5-xxl")
        text_encoder_t5 = T5EncoderModel(t5_config).to(dtype).to(device)
        text_encoder_t5.load_state_dict(load_file(os.path.join(MODEL_DIR, "clip", "t5xxl_fp8_e4m3fn.safetensors")))
        tokenizer_t5 = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        
        clip_config = CLIPTextConfig.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        clip_config.projection_dim = 768 
        text_encoder_clip = CLIPTextModelWithProjection(clip_config).to(dtype).to(device)
        
        clip_state_dict = load_file(os.path.join(MODEL_DIR, "clip", "ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors"))
        if "text_projection" in clip_state_dict:
            clip_state_dict["text_projection.weight"] = clip_state_dict.pop("text_projection")
        text_encoder_clip.load_state_dict(clip_state_dict, strict=False)
        tokenizer_clip = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")

        ae = AutoencoderKL.from_pretrained(os.path.join(MODEL_DIR, "sdxl-vae"), torch_dtype=dtype).to(device)
        
        unet = FluxUNet().to(device).to(dtype)
        unet.load_state_dict(load_file(os.path.join(MODEL_DIR, "unet", "flux1-fill-dev-fp8.safetensors")), strict=False)
    except Exception as e:
        print(f"\n--- Could not load a required model. --- \nError: {e}")
        return

    wrapped_unet = ModelWrapper(unet)
    tea_patcher = TeaCache()
    (patched_model_wrapper,) = tea_patcher.apply_teacache(
        model=wrapped_unet, model_type="flux", rel_l1_thresh=TEA_CACHE_THRESHOLD, 
        start_percent=TEA_CACHE_START_PERCENT, end_percent=TEA_CACHE_END_PERCENT, cache_device=device
    )

    img_cond = Image.open(INPUT_IMAGE_PATH).convert("RGB").resize((1024, 1024))
    img_cond = torch.from_numpy(np.array(img_cond) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device, dtype)

    mask = Image.open(MASK_IMAGE_PATH).convert("L").resize((1024, 1024))
    mask = torch.from_numpy(np.array(mask)).unsqueeze(0).unsqueeze(0).to(device, dtype) / 255.0

    with torch.no_grad():
        prompt_ids_t5 = tokenizer_t5(PROMPT, return_tensors="pt", max_length=77, padding="max_length", truncation=True).input_ids.to(device)
        prompt_embeds_t5 = text_encoder_t5(prompt_ids_t5)[0]
        # prompt_ids_clip = tokenizer_clip(PROMPT, return_tensors="pt", max_length=77, padding="max_length", truncation=True).input_ids.to(device)
        # prompt_embeds_clip = text_encoder_clip(prompt_ids_clip)[0]
        prompt_embeds = prompt_embeds_t5 # Corrected to use only T5
        img_cond_latents = ae.encode(img_cond * (1 - mask)).latent_dist.sample()
        
    num_steps = 30
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1).to(device)
    
    # This is a mock patch embedding to match shapes for the demo
    latents = torch.randn(1, (1024//8)**2, 384, device=device, dtype=dtype) * sigmas[0]
    
    patched_model_wrapper.model_options['transformer_options']['sample_sigmas'] = sigmas

    print("\n--- Starting Inference ---")
    for i in tqdm(range(num_steps)):
        sigma = sigmas[i]
        with torch.no_grad():
            main_call_kwargs = {
                "img": latents, "txt": prompt_embeds, "timesteps": torch.tensor([sigma * 1000], device=device),
                "img_ids": None, "txt_ids": None, "y": None, "transformer_options": patched_model_wrapper.model_options['transformer_options']
            }
            wrapper_kwargs = {
                "input": latents, "timestep": torch.tensor([sigma], device=device),
                "c": {"main_call_kwargs": main_call_kwargs, "transformer_options": patched_model_wrapper.model_options['transformer_options']}
            }
            model_output = patched_model_wrapper.model_unet_function(
                patched_model_wrapper.model.forward, wrapper_kwargs
            )

        d = (latents - model_output) / sigma
        latents = latents + d * (sigmas[i+1] - sigma)

    # This is a mock decoding process since the shapes do not match the VAE
    print("\n--- Mock Decoding ---")
    image_data = torch.randn(3, 1024, 1024).clamp(-1,1)
    image = ((image_data + 1) / 2).permute(1, 2, 0).cpu().numpy()
    
    Image.fromarray((image * 255).astype(np.uint8)).save(OUTPUT_IMAGE_PATH)
    print(f"\n--- Inference Finished ---\nOutput image saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_IMAGE_PATH) or not os.path.exists(MASK_IMAGE_PATH):
        print("Creating dummy input and mask images for first run...")
        Image.new('RGB', (1024, 1024), color = 'blue').save(INPUT_IMAGE_PATH)
        mask = Image.new('L', (1024, 1024), color = 'white')
        for x in range(256, 768):
            for y in range(256, 768): mask.putpixel((x, y), 0)
        mask.save(MASK_IMAGE_PATH)
    run_inference()

