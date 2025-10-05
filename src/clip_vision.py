import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from PIL import Image
import numpy as np
import math

class Output:
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, item):
        setattr(self, key, item)

def load_torch_file(ckpt_path, device="cpu"):
    with safe_open(ckpt_path, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    return state_dict

def clip_preprocess(image, size=384, mean_std=([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])):
    image = image.convert("RGB")
    w, h = image.size
    scale = size / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    
    left = (new_w - size) / 2
    top = (new_h - size) / 2
    right = (new_w + size) / 2
    bottom = (new_h + size) / 2
    image = image.crop((left, top, right, bottom))

    img_np = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

    mean = torch.tensor(mean_std[0])
    std = torch.tensor(mean_std[1])
    img_tensor = (img_tensor - mean[:, None, None]) / std[:, None, None]

    return img_tensor.unsqueeze(0)

ACTIVATIONS = { "gelu_pytorch_tanh": lambda a: F.gelu(a, approximate="tanh") }

class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(q.size(0), q.size(1), self.heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().reshape(x.size(0), x.size(1), -1)
        return self.out_proj(attn_output)

class CLIPMLP(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_size, activation, dtype, device):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate_size, bias=True, dtype=dtype, device=device)
        self.activation_fn = ACTIVATIONS[activation]
        self.fc2 = nn.Linear(intermediate_size, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return self.fc2(self.activation_fn(self.fc1(x)))

class CLIPLayer(torch.nn.Module):
    def __init__(self, config, dtype, device):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config["hidden_size"], eps=config.get("layer_norm_eps", 1e-6), dtype=dtype, device=device)
        self.self_attn = CLIPAttention(config["hidden_size"], config["num_attention_heads"], dtype, device)
        self.layer_norm2 = nn.LayerNorm(config["hidden_size"], eps=config.get("layer_norm_eps", 1e-6), dtype=dtype, device=device)
        self.mlp = CLIPMLP(config["hidden_size"], config["intermediate_size"], config["hidden_act"], dtype, device)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

class CLIPEncoder(torch.nn.Module):
    def __init__(self, config, dtype, device):
        super().__init__()
        self.layers = nn.ModuleList([CLIPLayer(config, dtype, device) for _ in range(config["num_hidden_layers"])])

    def forward(self, x, intermediate_output=None):
        all_hidden_states = ()
        for l in self.layers:
            if intermediate_output: all_hidden_states += (x,)
            x = l(x)
        all_hidden_states += (x,)
        return x, all_hidden_states

class CLIPVisionEmbeddings(torch.nn.Module):
    def __init__(self, config, dtype, device):
        super().__init__()
        num_patches = (config["image_size"] // config["patch_size"]) ** 2
        self.patch_embedding = nn.Conv2d(in_channels=config["num_channels"], out_channels=config["hidden_size"], kernel_size=config["patch_size"], stride=config["patch_size"], bias=True, dtype=dtype, device=device)
        self.position_embedding = nn.Embedding(num_patches, config["hidden_size"], dtype=dtype, device=device)
        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)), persistent=False)

    def forward(self, pixel_values):
        embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        pos_embeds = self.position_embedding(self.position_ids)
        return embeds + pos_embeds

class CLIPVision(torch.nn.Module):
    def __init__(self, config, dtype, device):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(config, dtype, device)
        self.pre_layrnorm = nn.Identity()
        self.encoder = CLIPEncoder(config, dtype, device)
        self.post_layernorm = nn.LayerNorm(config["hidden_size"], dtype=dtype, device=device)

    def forward(self, pixel_values, intermediate_output=None):
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)
        x, i = self.encoder(x, intermediate_output=intermediate_output)
        x = self.post_layernorm(x)
        return x, i, x

class CLIPVisionModelProjection(torch.nn.Module):
    def __init__(self, config, dtype, device):
        super().__init__()
        self.vision_model = CLIPVision(config, dtype, device)
        if config.get("projection_dim") is not None:
            self.visual_projection = nn.Linear(config["hidden_size"], config["projection_dim"], bias=False, dtype=dtype, device=device)
        else:
            self.visual_projection = nn.Identity()

    def forward(self, pixel_values, intermediate_output=None):
        vision_outputs = self.vision_model(pixel_values, intermediate_output=intermediate_output)
        pooled_output = vision_outputs[2]
        projected_output = self.visual_projection(pooled_output)
        return (vision_outputs[0], vision_outputs[1], projected_output, None)

class StandaloneClipVisionModel:
    def __init__(self, config, device, dtype):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.model = CLIPVisionModelProjection(config, dtype, device)
        self.model.eval()

    def load_sd(self, state_dict):
        if "proj" in state_dict:
            state_dict["visual_projection.weight"] = state_dict.pop("proj").transpose(0, 1)
        filtered_sd = {k: v for k, v in state_dict.items() if not k.startswith("vision_model.head.")}
        self.model.load_state_dict(filtered_sd, strict=True)

    @torch.no_grad()
    def encode_image(self, image_input):
        if isinstance(image_input, str):
            pil_image = Image.open(image_input)
        else:
            pil_image = image_input

        image_tensor = clip_preprocess(pil_image, size=self.config["image_size"], mean_std=(self.config["image_mean"], self.config["image_std"]))
        pixel_values = image_tensor.to(device=self.device, dtype=self.dtype)
        last_hidden_state, all_h_states, projected_embeds, _ = self.model(pixel_values, intermediate_output=True)
        outputs = Output()
        all_hidden_states = torch.stack(all_h_states)
        outputs["last_hidden_state"] = last_hidden_state
        outputs["penultimate_hidden_states"] = all_hidden_states[-2]
        outputs["all_hidden_states"] = all_hidden_states
        outputs["image_embeds"] = projected_embeds
        return outputs

def load(ckpt_path):
    state_dict = load_torch_file(ckpt_path)
    if state_dict is None: return None

    is_siglip_384 = ("vision_model.encoder.layers.26.mlp.fc2.weight" in state_dict)
    if not is_siglip_384: return None

    model_config = {
        "num_channels": 3, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1152,
        "image_size": 384, "intermediate_size": 4304, "model_type": "siglip_vision_model",
        "num_attention_heads": 16, "num_hidden_layers": 27, "patch_size": 14,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "layer_norm_eps": 1e-6
    }
    
    if "visual_projection.weight" in state_dict or "proj" in state_dict:
        print("INFO: Visual projection layer weights found in file. Creating model with projection.")
        model_config["projection_dim"] = 1152
    else:
        print("INFO: No visual projection layer weights found. Creating model without projection.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if "cuda" in str(device) else torch.float32
    clip_vision_model = StandaloneClipVisionModel(config=model_config, device=device, dtype=dtype)
    clip_vision_model.load_sd(state_dict)
    
    return clip_vision_model