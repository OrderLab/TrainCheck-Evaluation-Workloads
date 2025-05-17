import transformers.modeling_flash_attention_utils
from transformers.models.m2m_100.modeling_m2m_100 import M2M100FlashAttention2
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import transformers
import torch

import traincheck
from traincheck import annotate_stage
traincheck.instrumentor.tracer.DISABLE_WRAPPER = True

# Load a model and tokenizer
model_name = "facebook/m2m100_418M"
annotate_stage("init")  # ML_DAIKON: stage annotation

# Initialize the model with the modified configuration
model = M2M100ForConditionalGeneration.from_pretrained(model_name, attn_implementation="flash_attention_2")
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
print("attention_implementations:", model.config._attn_implementation)

traincheck.instrumentor.tracer.DISABLE_WRAPPER = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).bfloat16()

annotate_stage("testing")  # ML_DAIKON: stage annotation
# Set the model to evaluation mode
model.eval()

# Define a flag to confirm FA2 usage
fa2_used = False

# Register a forward hook on FlashAttention2 to confirm it’s being used
def hook_fn(module, input, output):
    global fa2_used
    fa2_used = True

# Attach hook to each instance of FlashAttention2 in the model
for module in model.modules():
    if isinstance(module, M2M100FlashAttention2):
        module.register_forward_hook(hook_fn)

# Prepare inputs
input_text = "This is a test sentence."

traincheck.instrumentor.tracer.DISABLE_WRAPPER = True
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
traincheck.instrumentor.tracer.DISABLE_WRAPPER = False

decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = torch.tensor([[decoder_start_token_id]]).to(device)

# Run the model multiple times in evaluation mode and store outputs
outputs = []
for _ in range(3):
    # Run the model in evaluation mode with mixed precision
    with torch.no_grad(), torch.cuda.amp.autocast():  # Enables mixed precision
        output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits
        outputs.append(output)

# Check if FlashAttention2 was used
if fa2_used:
    print("Confirmed: FlashAttention2 was used during the forward pass.")
else:
    print("Warning: FlashAttention2 was not used in the forward pass.")

# Check if all outputs are the same, which they should be in evaluation mode
all_equal = all(torch.allclose(outputs[i], outputs[i + 1], atol=1e-5) for i in range(len(outputs) - 1))

# Print result
if all_equal:
    print("Bug not exposed: Outputs are consistent across runs in evaluation mode.")
else:
    print("Bug exposed: Outputs vary across runs in evaluation mode due to dropout not being disabled!")
