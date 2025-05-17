# TF-34204

https://github.com/huggingface/transformers/issues/34204

## Environment Setup
- transformers version: 4.45.2
- Python version: 3.10.15
- PyTorch version (GPU?): 2.2.0+cu121 (True)

## What to expect

The PixtralProcessor should process all inputs in the batch and return:
- input_ids with shape [50, sequence_length].
- pixel_values with shape [50, channels, height, width].
## What is the bug

### Symptoms
- Input IDs: The shape of batch["input_ids"] is [1, sequence_length] instead of [50, sequence_length].

## Root Causes
Incorrect handling of list aggregation aggregating images into a single list.
[details in original issue link](https://github.com/huggingface/transformers/issues/34204)


## TODO: Convert to a e2e real world example:
https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl