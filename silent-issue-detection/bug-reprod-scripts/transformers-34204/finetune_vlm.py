import torch
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, PixtralProcessor, Qwen2VLProcessor
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

dataset_id = "HuggingFaceM4/ChartQA"
model_id = "mistral-community/pixtral-12b"

# Configure training arguments
training_args = SFTConfig(
    output_dir="pixtral-7b-traincheck-buggy",  # Directory to save the model
    num_train_epochs=5,  # Number of training epochs
    per_device_train_batch_size=3,  # Batch size for training 
    per_device_eval_batch_size=4,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_steps=10,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=20,  # Steps interval for saving
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    load_best_model_at_end=True,  # Load the best model after training
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    # tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    # push_to_hub=True,  # Whether to push model to Hugging Face Hub
    report_to="tensorboard",  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    # max_seq_length=1024  # Maximum sequence length for input
)


system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"][0]}],
        },
    ]

train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=["train[:1%]", "val[:1%]", "test[:1%]"]) # for it to run through quickly

train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]


def apply_chat_template(example):
    """
    Custom function to convert example into a formatted text template.
    
    Args:
    - example (list): A list of dictionaries representing system, user, and assistant messages.
    
    Returns:
    - str: A formatted string with the content of the example.
    """
    template = ""

    for entry in example:
        role = entry['role']
        content_list = entry['content']
        
        # Add the role header
        if role == 'system':
            template += "<s>[SYSTEM]\n"
        elif role == 'user':
            template += "[USER]\n"
        elif role == 'assistant':
            template += "[ASSISTANT]\n"

        # Add the content
        for content in content_list:
            content_type = content['type']
            
            if content_type == 'text':
                template += content['text'] + "\n"
            elif content_type == 'image':
                # Replace image with a placeholder
                template += "[IMG]\n"

        # Close the role if applicable
        template += "[/SYSTEM]\n" if role == 'system' else "[/USER]\n" if role == 'user' else "[/ASSISTANT]\n"

    return template

print(apply_chat_template(train_dataset[0]))
processor = PixtralProcessor.from_pretrained(model_id)
if processor.tokenizer.pad_token is None:
    if processor.tokenizer.eos_token is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    else:
        processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb_config)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [
        apply_chat_template(example) for example in examples
    ]  # Prepare texts for processing

    # bug free if image_inputs = [[img_example_0], [img_example_1], ...]
    # image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # buggy if image_inputs = [img_example_0, img_example_1, ...]
    image_inputs = [process_vision_info(example)[0][0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(
        image_inputs, texts, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch


# batch = collate_fn(train_dataset[0:4])

# print("Batch keys:", batch.keys())
# print("Input IDs shape:", len(batch["input_ids"]))     # this should be 50, but found 1 instead
# print("Pixel values shape:", len(batch["pixel_values"][0]))

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)

trainer.train()

# IMG_URLS = [
#      "https://picsum.photos/id/237/400/300",
#      "https://picsum.photos/id/231/200/300",
#      "https://picsum.photos/id/27/500/500",
#      "https://picsum.photos/id/17/150/600",
# ]
# PROMPT = "<s>[INST]Describe the images.\n[IMG][IMG][IMG][IMG][/INST]"

# inputs = processor(text=PROMPT, images=IMG_URLS, return_tensors="pt").to("cuda")
# generate_ids = model.generate(**inputs, max_new_tokens=500)
# output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(output)
