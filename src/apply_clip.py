import torch
from transformers import AutoProcessor, SiglipVisionModel
from PIL import Image
import requests

CLIP_VIS_MODEL_PATH = "/teamspace/studios/this_studio/.porting/models/siglip_local"
IMAGE_PATH = "/teamspace/studios/this_studio/.porting/imgs/input_img_1.jpg"
device = "cuda" if torch.cuda.is_available() else "cpu"

def clip_process():

    try:   
        processor = AutoProcessor.from_pretrained(CLIP_VIS_MODEL_PATH)
        model = SiglipVisionModel.from_pretrained(CLIP_VIS_MODEL_PATH).to(device)
        model.eval()
        print("Model and processor loaded successfully.")
        return processor, model
    except OSError:
        print(f"Error: Could not find a valid model directory at '{CLIP_VIS_MODEL_PATH}'.")
        print("Please ensure you have run the download script first and the directory exists.")
        exit()

def run_clip(image_path = IMAGE_PATH):
    try:
        processor, model = clip_process()

        image = Image.open(image_path).convert("RGB")
        print("\nEncoding a sample image...")

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            clip_vision_output = outputs.last_hidden_state
            # image_embeds = outputs.pooler_output

        print("--- Success! ---")
        print(f"Shape of the final image embedding tensor: {clip_vision_output.shape}")
        return clip_vision_output
    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")

if __name__ == "__main__":
    out = run_clip()