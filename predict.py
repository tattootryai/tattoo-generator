import os
import torch
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        lora_path = "https://huggingface.co/tattootryai/tattoo-pro-lora/resolve/main/tattoo_pro.safetensors"

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to("cuda")

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights(lora_path)

    def predict(
        self,
        prompt: str = Input(description="Tattoo prompt"),
        negative_prompt: str = Input(default="ugly, blurry, deformed, extra limbs", description="What to avoid"),
        guidance_scale: float = Input(default=7.5),
        num_inference_steps: int = Input(default=30),
        width: int = Input(default=768),
        height: int = Input(default=768),
        seed: int = Input(default=None)
    ) -> Path:

        generator = torch.manual_seed(seed) if seed else None

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        ).images[0]

        output_path = "/tmp/output.png"
        image.save(output_path)
        return Path(output_path)
