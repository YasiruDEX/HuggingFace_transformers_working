import torch
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np

# Load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each expert (80/20) here
n_steps = 40
high_noise_frac = 0.8

def generate_image(prompt):
    # Run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    return image

def display_image(image):
    # Convert the PIL image to a numpy array for plotting
    image_np = np.array(image)
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

def main():
    while True:
        prompt = input("Enter a prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        image = generate_image(prompt)
        display_image(image)

if __name__ == "__main__":
    main()
