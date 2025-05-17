from transformers import AutoProcessor
from PIL import Image

processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")

processor.tokenizer.pad_token = processor.tokenizer.eos_token


prompts = [f"{x}" for x in range(50)]
print(f"prompts: {prompts}")

images = [Image.new("RGB", (224, 224), color="white") for _ in range(50)]
print(f"len images: {len(images)}")

batch = processor(images, prompts, padding=True, return_tensors="pt")

print("Batch keys:", batch.keys())
print("Input IDs shape:", len(batch["input_ids"]))     # this should be 50, but found 1 instead
print("Pixel values shape:", len(batch["pixel_values"][0]))
print("Input IDs:", batch["input_ids"])     # bug: only the first token is in input_ids


