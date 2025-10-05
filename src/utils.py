import os
import torch
from safetensors.torch import load_file
from copy import deepcopy
import re
from collections import defaultdict
import safetensors
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORAS_TO_APPLY = {
        "comfyui_subject_lora16.safetensors": 1.0,
    }

def load_flux_fill():
    model_path = os.path.join(BASE_DIR, "models", "unet", "flux1-fill-dev-fp8.safetensors")
    print(f"Loading model: {os.path.basename(model_path)}")
    return load_file(model_path)


def load_torch_file(ckpt, safe_load=False, device="cpu"):
    """
    Loads a torch file, handling both safetensors and standard PyTorch pickle files.
    """
    if safe_load:
        sd = {}
        with safetensors.safe_open(ckpt, framework="pt", device=device) as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
        return sd
    else:
        pl_sd = torch.load(ckpt, map_location=device, weights_only=False)
        return pl_sd

def zero_out(conditioning):
    c = []
    for t in conditioning:
        d = t[1].copy()
        pooled_output = d.get("pooled_output", None)
        if pooled_output is not None:
            d["pooled_output"] = torch.zeros_like(pooled_output)
        conditioning_lyrics = d.get("conditioning_lyrics", None)
        if conditioning_lyrics is not None:
            d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
        n = [torch.zeros_like(t[0]), d]
        c.append(n)
    return (c, )

def apply_flux_guidance(conditioning, guidance):

    if guidance == 1.0:
        return (conditioning,)

    c = []
    for t in conditioning:
        d = t[1].copy()
        d['guidance'] = guidance
        new_item = [t[0], d]
        c.append(new_item)
    return (c,)
def translate_lora_key_to_model_key(lora_key: str) -> str:
    base_lora_key = re.sub(r"\.(alpha|lora_down|lora_up).*", "", lora_key)

    if not base_lora_key.startswith("lora_unet_"):
        raise ValueError(f"Unexpected LoRA key format: {lora_key}")

    key_body = base_lora_key[len("lora_unet_"):]
    parts = key_body.split("_")

    if parts[0] == "double" and parts[1] == "blocks":
        block_id = parts[2]
        if len(parts) < 5:
            raise ValueError(f"Unexpected double_block LoRA key format: {lora_key}")
        
        component = f"{parts[3]}_{parts[4]}"
        sub_path_parts = parts[5:]
        
        final_path = ".".join([component] + sub_path_parts)
        return f"double_blocks.{block_id}.{final_path}"

    if parts[0] == "single" and parts[1] == "blocks":
        block_id = parts[2]
        sub_path = ".".join(parts[3:])
        return f"single_blocks.{block_id}.{sub_path}"

    raise ValueError(f"Unexpected LoRA key format: {lora_key}")

def apply_lora_to_model(model_state_dict, lora_state_dict, strength: float):
    loras = defaultdict(dict)
    for key, tensor in lora_state_dict.items():
        if "lora_down" in key:
            base_key = key.replace(".lora_down.weight", "")
            loras[base_key]["down"] = tensor
        elif "lora_up" in key:
            base_key = key.replace(".lora_up.weight", "")
            loras[base_key]["up"] = tensor
        elif "alpha" in key:
            base_key = key.replace(".alpha", "")
            loras[base_key]["alpha"] = tensor

    applied_count = 0
    skipped_count = 0

    for base_lora_key, components in loras.items():
        model_key_prefix = translate_lora_key_to_model_key(base_lora_key)
        
        lora_down = components.get("down")
        lora_up = components.get("up")
        alpha = components.get("alpha")

        if lora_down is None or lora_up is None:
            print(f"[SKIPPED] Incomplete LoRA components for {base_lora_key}")
            skipped_count += 1
            continue

        model_key_weight = f"{model_key_prefix}.weight"
        model_key_bias = f"{model_key_prefix}.bias"

        target_key = None
        if model_key_weight in model_state_dict:
            target_key = model_key_weight
        elif model_key_bias in model_state_dict:
            target_key = model_key_bias

        if target_key:
            target_tensor = model_state_dict[target_key]
            
            if alpha is not None:
                scale = alpha.item() / lora_down.shape[0]
            else:
                scale = 1.0

            lora_weight = (lora_up @ lora_down).to(torch.float32) * scale * strength
            
            if target_tensor.shape != lora_weight.shape:
                print(f"[SKIPPED] Shape mismatch for {target_key}: model is {target_tensor.shape}, LoRA is {lora_weight.shape}")
                skipped_count += 1
                continue
                
            target_tensor.data += lora_weight.to(target_tensor.dtype)
            applied_count += 1
            print(f"[APPLIED] {base_lora_key} -> {target_key}")
        else:
            print(f"[SKIPPED] {base_lora_key} -> {model_key_prefix} (no match in model)")
            skipped_count += 1

    print(f"\n-> Applied {applied_count} LoRA layers and skipped {skipped_count}\n")

def load_loras():
    base_weights = load_flux_fill()
    for lora_name, strength in LORAS_TO_APPLY.items():
        lora_path = os.path.join(BASE_DIR, "models", "loras", lora_name)
        if not os.path.exists(lora_path):
            print(f"Warning: LoRA file not found at {lora_path}, skipping.")
            continue
        
        print(f"Loading LoRA: {lora_name} with strength {strength}")
        lora_weights = load_file(lora_path)
        apply_lora_to_model(base_weights, lora_weights, strength)
        
    return base_weights

if __name__ == "__main__":
    print("\n--- Script Started ---\n")
    updated_weights = load_torch_file(os.path.join(BASE_DIR, "models", "unet", "flux1-fill-dev-fp8.safetensors"), safe_load=True)
    print("\n--- Script Finished ---")