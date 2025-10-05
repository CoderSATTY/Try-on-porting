import torch
import numpy as np
from PIL import Image
from safetensors import safe_open
from utils import load_torch_file
from style_utils import StyleModel, ReduxImageEncoder
import clip_vision as cvis
from apply_clip import run_clip
import joblib 

CLIP_MODEL_PATH = "/teamspace/studios/this_studio/.porting/models/clip_vision/sigclip_vision_patch14_384.safetensors"
STYLE_MODEL_PATH = "/teamspace/studios/this_studio/.porting/models/style_models/flux1-redux-dev.safetensors"

class CLIPOutputWrapper:
    def __init__(self, tensor):
        self.last_hidden_state = tensor

def load_style_model(ckpt_path):
    style_path = ckpt_path
    model_data = load_torch_file(style_path, safe_load=True)
    model = ReduxImageEncoder()
    model.load_state_dict(model_data)
    return StyleModel(model)


def apply_stylemodel(conditioning, style_model, clip_vision_output, strength, strength_type):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if strength_type == "multiply":
            cond *= strength

        n = cond.shape[1]
        c_out = []
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()
            # even if the strength is 1.0 (i.e, no change), if there's already a mask, we have to add to it
            if "attention_mask" in keys or (strength_type == "attn_bias" and strength != 1.0):
                # math.log raises an error if the argument is zero
                # torch.log returns -inf, which is what we want
                attn_bias = torch.log(torch.Tensor([strength if strength_type == "attn_bias" else 1.0]))
                # get the size of the mask image
                mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                n_ref = mask_ref_size[0] * mask_ref_size[1]
                n_txt = txt.shape[1]
                # grab the existing mask
                mask = keys.get("attention_mask", None)
                # create a default mask if it doesn't exist
                if mask is None:
                    mask = torch.zeros((txt.shape[0], n_txt + n_ref, n_txt + n_ref), dtype=torch.float16)
                # convert the mask dtype, because it might be boolean
                # we want it to be interpreted as a bias
                if mask.dtype == torch.bool:
                    # log(True) = log(1) = 0
                    # log(False) = log(0) = -inf
                    mask = torch.log(mask.to(dtype=torch.float16))
                # now we make the mask bigger to add space for our new tokens
                new_mask = torch.zeros((txt.shape[0], n_txt + n + n_ref, n_txt + n + n_ref), dtype=torch.float16)
                # copy over the old mask, in quandrants
                new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
                new_mask[:, :n_txt, n_txt+n:] = mask[:, :n_txt, n_txt:]
                new_mask[:, n_txt+n:, :n_txt] = mask[:, n_txt:, :n_txt]
                new_mask[:, n_txt+n:, n_txt+n:] = mask[:, n_txt:, n_txt:]
                # now fill in the attention bias to our redux tokens
                new_mask[:, :n_txt, n_txt:n_txt+n] = attn_bias
                new_mask[:, n_txt+n:, n_txt:n_txt+n] = attn_bias
                keys["attention_mask"] = new_mask.to(txt.device)
                keys["attention_mask_img_shape"] = mask_ref_size

            c_out.append([torch.cat((txt, cond), dim=1), keys])

        return (c_out,)


if __name__ == "__main__":

    clip_vis_tensor = run_clip()
    clip_vis_output = CLIPOutputWrapper(clip_vis_tensor)
    style_model = load_style_model(STYLE_MODEL_PATH)
    conditioning = joblib.load("/teamspace/studios/this_studio/.porting/models/conditioning/text_cache.conditioning")

    out_cond, = apply_stylemodel(conditioning, style_model, clip_vis_output, 1, "multiply")
    print("Output conditioning shape:",out_cond)
    print("Successfully applied the style model.")
