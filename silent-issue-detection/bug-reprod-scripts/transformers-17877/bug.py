import torch
import math
from transformers import GPT2Model, GPT2Config

# Dictionary to keep track of tensor initialization counts
initialization_counts = {}
normal_call_counts = {}


# Monkey patch the normal_ function to count its usage
original_normal_ = torch.Tensor.normal_

import functools
@functools.wraps(original_normal_)
def custom_normal_(self, mean=0.0, std=1.0, generator=None):
    param_id = id(self)
    if param_id not in normal_call_counts:
        normal_call_counts[param_id] = 0
    normal_call_counts[param_id] += 1
    # return original_normal_(self, mean, std, generator)
    pass # NOTE: could not call the original function here, otherwise it would expose with the following runtime error:
    # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
    # Reason: the original function is in-place and would modify the tensor in-place, which is not allowed in PyTorch

torch.Tensor.normal_ = custom_normal_

# Create GPT2 configuration and model
configuration = GPT2Config()
model = GPT2Model(configuration)

# Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme
for name, p in model.named_parameters():
    if "c_proj" in name and "weight" in name:
        # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
        print(id(p), name)
        custom_normal_(p, mean=0.0, std=(configuration.initializer_range / math.sqrt(2 * configuration.n_layer)))



# Dummy input to trigger the forward pass
input_ids = torch.tensor([[1, 2, 3, 4, 5]])

# Perform a forward pass to activate the hooks
with torch.no_grad():
    model(input_ids)

# Print the count of normal_ calls for each parameter
for param_id, info in normal_call_counts.items():
    print(f"Parameter: {param_id} | Normal_ Call Count: {info}")