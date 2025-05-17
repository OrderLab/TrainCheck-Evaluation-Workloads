import transformers.modeling_flash_attention_utils
from transformers.models.opt.modeling_opt import OptFlashAttention2
from transformers import OPTForCausalLM, AutoTokenizer
import transformers
import torch

import mldaikon
from mldaikon import annotate_stage

mldaikon.instrumentor.tracer.DISABLE_WRAPPER = True

model_name = "facebook/opt-350m"
model = OPTForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fa2_used = False

mldaikon.instrumentor.tracer.DISABLE_WRAPPER = False


def check_attention_layers(model):
    for name, module in model.named_modules():
        print(name, type(module))


def convert_attention_layers_to_half(model):
    for name, module in model.named_modules():
        if "attention" in name or "attn" in name or "flash" in name:
            module.half()


def hook_fn(module, input, output):
    global fa2_used
    fa2_used = True


def train(model, optimizer, epoch=2, input_ids=None, labels=None):
    annotate_stage("training")  # ML_DAIKON: stage annotation
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for _ in range(epoch):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()


def main():
    # check_attention_layers(model)
    annotate_stage("init")  # ML_DAIKON: stage annotation
    print("Attention implementation:", model.config._attn_implementation)
    convert_attention_layers_to_half(model)
    for module in model.modules():
        if isinstance(module, OptFlashAttention2):
            module.dropout = 0.1
            module.register_forward_hook(hook_fn)

    global fa2_used
    fa2_used = False
    model.to(device).bfloat16()
    model.train()

    input_text = "This is a test sentence for OPT model."

    mldaikon.instrumentor.tracer.DISABLE_WRAPPER = True
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    mldaikon.instrumentor.tracer.DISABLE_WRAPPER = False
    labels = input_ids.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    fa2_used = False
    train(model, optimizer, input_ids=input_ids, labels=labels)

    if fa2_used:
        print("Confirmed Training: FlashAttention2 was used during the forward pass.")
    else:
        print("Warning Training: FlashAttention2 was not used in the forward pass.")

    annotate_stage("testing")  # ML_DAIKON: stage annotation
    model.eval()
    fa2_used = False

    outputs = []
    for _ in range(3):
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model(input_ids=input_ids).logits
            outputs.append(output)

    if fa2_used:
        print("Confirmed Eval: FlashAttention2 was used during the forward pass.")
    else:
        print("Warning Eval: FlashAttention2 was not used in the forward pass.")

    all_equal = all(torch.allclose(outputs[i], outputs[i + 1], atol=1e-5) for i in range(len(outputs) - 1))

    if all_equal:
        print("Bug not exposed: Outputs are consistent across runs in evaluation mode.")
    else:
        print("Bug exposed: Outputs vary across runs in evaluation mode due to dropout not being disabled!")


if __name__ == "__main__":
    main()
