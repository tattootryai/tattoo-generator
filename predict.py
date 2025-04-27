import torch
from diffusers import StableDiffusionPipeline

def load_model():
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    lora_path = "https://huggingface.co/tattootryai/TattooXL-LoRA/resolve/main/tattooxl_lora.safetensors"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.load_lora_weights(lora_path)

    return pipe

model = load_model()

def predict(prompt: str = "a blackwork eagle head tattoo on forearm"):
    image = model(prompt).images[0]
    image.save("output.png")
    return image
