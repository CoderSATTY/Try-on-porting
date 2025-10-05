import torch
import numpy as np
import os
import math
from PIL import Image
from einops import rearrange, repeat
from numpy import poly1d
from unittest.mock import patch, MagicMock

def create_dummy_inpainting_data(size=(512, 512)):
    img = Image.new('RGB', size, color='red')
    img.save("img_cond.png")
    mask = Image.new('L', size, color='white')
    for x in range(size[0] // 4, 3 * size[0] // 4):
        for y in range(size[1] // 4, 3 * size[1] // 4):
            mask.putpixel((x, y), 0)
    mask.save("mask.png")
    return "img_cond.png", "mask.png"

def apply_mod(tensor, m_mult, m_add=None, modulation_dims=None):
    if modulation_dims is None:
        if m_add is not None:
            return torch.addcmul(m_add, tensor, m_mult)
        else:
            return tensor * m_mult
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
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

def teacache_flux_forward(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor = None,
        control=None,
        transformer_options={},
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
    rel_l1_thresh = transformer_options.get("rel_l1_thresh")
    coefficients = transformer_options.get("coefficients")
    enable_teacache = transformer_options.get("enable_teacache", True)
    cache_device = transformer_options.get("cache_device")

    if y is None:
        y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
        
    img_unprocessed = img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
    txt = self.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    img_mod1, _ = self.double_blocks[0].img_mod(vec)
    modulated_inp = self.double_blocks[0].img_norm1(img)
    modulated_inp = apply_mod(modulated_inp, (1 + img_mod1.scale), img_mod1.shift).to(cache_device)

    if not hasattr(self, 'accumulated_rel_l1_distance'):
        should_calc = True
        self.accumulated_rel_l1_distance = 0
    else:
        try:
            dist = ((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean())
            self.accumulated_rel_l1_distance += poly1d(coefficients, dist.item()).abs()
            if self.accumulated_rel_l1_distance < rel_l1_thresh:
                should_calc = False
                print("   -> [TeaCache] SKIP: Accumulated change is below threshold.")
            else:
                should_calc = True
                print("   -> [TeaCache] COMPUTE: Threshold exceeded. Resetting accumulator.")
                self.accumulated_rel_l1_distance = 0
        except Exception as e:
            should_calc = True
            self.accumulated_rel_l1_distance = 0

    self.previous_modulated_input = modulated_inp

    if not enable_teacache:
        should_calc = True

    if not should_calc:
        img = img_unprocessed + self.previous_residual.to(img.device)
    else:
        ori_img = img_unprocessed.clone().to(cache_device)
        img = self.forward_orig(img=img_unprocessed, img_ids=img_ids, txt=txt, txt_ids=txt_ids, timesteps=timesteps, y=y, guidance=guidance, control=control, transformer_options=transformer_options, attn_mask=attn_mask)
        self.previous_residual = img.clone().to(cache_device) - ori_img
    
    return img

SUPPORTED_MODELS_COEFFICIENTS = {
    "flux-fill": [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
}

class TeaCache:
    def apply_teacache(self, model, model_type: str, rel_l1_thresh: float, cache_device: str):
        if rel_l1_thresh == 0:
            return model

        new_model = model
        new_model.model_options = {
            'transformer_options': {
                "rel_l1_thresh": rel_l1_thresh,
                "coefficients": SUPPORTED_MODELS_COEFFICIENTS[model_type],
                "cache_device": torch.device(cache_device)
            }
        }

        diffusion_model = new_model
        diffusion_model.forward_orig = diffusion_model.forward
        diffusion_model.forward = teacache_flux_forward.__get__(diffusion_model, diffusion_model.__class__)
        
        print("Model has been patched with TeaCache.")
        return new_model

def prepare_fill(t5, clip, img, prompt, ae, img_cond_path, mask_path) -> dict[str, torch.Tensor]:
    bs, _, _, _ = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = torch.from_numpy(mask).float() / 255.0
    mask = rearrange(mask, "h w -> 1 1 h w")

    with torch.no_grad():
        img_cond = img_cond.to(img.device)
        mask = mask.to(img.device)
        img_cond = img_cond * (1 - mask)
        img_cond = ae.encode(img_cond)
        mask = mask[:, 0, :, :]
        mask = mask.to(torch.bfloat16)
        mask = rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w", ph=8, pw=8)
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if mask.shape[0] == 1 and bs > 1:
            mask = repeat(mask, "1 ... -> bs ...", bs=bs)

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)
    
    final_img_cond = torch.cat((img_cond, mask), dim=-1)

    return_dict = {
        "txt": torch.randn(1, 1024, 4096, device=img.device),
        "img": torch.randn(1, 1024, 4096, device=img.device),
        "txt_ids": torch.randn(1, 1024, device=img.device),
        "img_ids": torch.randn(1, 1024, device=img.device),
        "y": torch.randn(1, 256, device=img.device),
        "img_cond": final_img_cond.to(img.device)
    }
    return return_dict

if __name__ == "__main__":
    print("--- Script Started: Applying TeaCache to FLUX-Fill Inpainting ---\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_path, mask_path = create_dummy_inpainting_data()
    prompt = "A majestic cat sitting on a throne"
    start_percent, end_percent = 0.0, 1.0

    with patch('__main__.Image.open'), patch('__main__.torch.from_numpy') as mock_from_numpy:
        mock_from_numpy.side_effect = [
            torch.randn(512, 512, 3, device=device),
            torch.randn(512, 512, device=device)
        ]

        flux_model = MagicMock()
        flux_model.params = type('obj', (object,), {'vec_in_dim': 256, 'guidance_embed': False})
        flux_model.img_in = torch.nn.Identity().to(device)
        flux_model.time_in = torch.nn.Identity().to(device)
        flux_model.vector_in = torch.nn.Identity().to(device)
        flux_model.txt_in = torch.nn.Identity().to(device)
        flux_model.pe_embedder = torch.nn.Identity().to(device)
        
        mock_block = MagicMock()
        img_mod1_mock = MagicMock()
        img_mod1_mock.scale = torch.randn(1, 1, device=device)
        img_mod1_mock.shift = torch.randn(1, 1, device=device)
        mock_block.img_mod.return_value = (img_mod1_mock, None)
        
        def dynamic_norm_return(*args, **kwargs):
            return torch.randn(1, 1024, 4096, device=device)
        mock_block.img_norm1.side_effect = dynamic_norm_return
        
        flux_model.double_blocks = [mock_block]
        flux_model.single_blocks = [MagicMock()]
        flux_model.final_layer = MagicMock()
        
        def mock_forward_pass(*args, **kwargs):
            # Simulate the model output being slightly different each time
            return kwargs['img'] + torch.randn_like(kwargs['img']) * 0.1
        flux_model.forward.side_effect = mock_forward_pass
        flux_model.to.return_value = flux_model

        t5, clip, ae = MagicMock(), MagicMock(), MagicMock()
        ae.encode.return_value = torch.randn(1, 4, 64, 64, device=device)
        
        tea_patcher = TeaCache()
        patched_model = tea_patcher.apply_teacache(
            model=flux_model, model_type="flux-fill", rel_l1_thresh=0.4, cache_device=device
        )

        print("\nPreparing inpainting conditioning data...")
        start_image_tensor = torch.randn(1, 3, 512, 512, device=device)
        conditioning_data = prepare_fill(t5, clip, start_image_tensor, prompt, ae, img_path, mask_path)
        print("Conditioning data prepared.\n")

        print("--- Starting Simulated Sampling Loop ---")
        num_steps = 10
        sigmas = torch.linspace(1, 0, num_steps + 1).to(device)
        latents = torch.randn(1, 1024, 4096, device=device)

        for i in range(num_steps):
            current_sigma = sigmas[i]
            print(f"Step {i+1}/{num_steps} (Sigma: {current_sigma.item():.2f})")
            
            current_percent = i / (num_steps - 1) if num_steps > 1 else 1.0
            enable_teacache = start_percent <= current_percent <= end_percent
            patched_model.model_options['transformer_options']['enable_teacache'] = enable_teacache
            print(f"   -> TeaCache Enabled: {enable_teacache}")

            output = patched_model.forward(
                img=latents,
                img_ids=conditioning_data["img_ids"],
                txt=conditioning_data["txt"],
                txt_ids=conditioning_data["txt_ids"],
                timesteps=current_sigma.unsqueeze(0),
                y=conditioning_data["y"],
                transformer_options=patched_model.model_options['transformer_options']
            )
            latents = output

    print("\n--- Simulated Sampling Loop Finished ---")
    print("\n--- Script Finished ---")

    os.remove(img_path)
    os.remove(mask_path)

