from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO

# 1. Load the processor and model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

prompts = [f"{x}" for x in range(50)]
print(f"prompts: {prompts}")

images = [Image.new("RGB", (224, 224), color="white") for _ in range(50)]
print(f"len images: {len(images)}")


# 4. Use CLIPProcessor to preprocess images and prompts
batch = processor(text=prompts, images=images, return_tensors="pt", padding=True)

# 5. Inspect the batch
print("Batch keys:", batch.keys())
print("Input IDs shape:", batch["input_ids"].shape)  # Should match the number of prompts
print("Pixel values shape:", batch["pixel_values"].shape)  # Should match the number of images
print("Input IDs (first 3):", batch["input_ids"][:3])  # Print tokenized prompts (first 3)

