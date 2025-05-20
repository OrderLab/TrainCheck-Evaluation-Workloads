import transformers.modeling_flash_attention_utils
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/home/yuxuan/TrainCheck-Evaluation-Workloads/silent-issue-detection/bug-reprod-scripts/transformers-33844/traincheck_run_bug_torch_2.2.2+cu121_transformers_4.45.0_2025-05-19_15-58-03"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(transformers.modeling_flash_attention_utils, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from transformers.models.m2m_100.modeling_m2m_100 import M2M100FlashAttention2
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(M2M100FlashAttention2, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(M2M100ForConditionalGeneration, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(M2M100Tokenizer, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import transformers
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(transformers, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import traincheck
from traincheck import annotate_stage
traincheck.instrumentor.tracer.DISABLE_WRAPPER = True
model_name = 'facebook/m2m100_418M'
annotate_stage('init')
model = M2M100ForConditionalGeneration.from_pretrained(model_name, attn_implementation='flash_attention_2')
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
print('attention_implementations:', model.config._attn_implementation)
traincheck.instrumentor.tracer.DISABLE_WRAPPER = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).bfloat16()
annotate_stage('testing')
model.eval()
fa2_used = False

def hook_fn(module, input, output):
    global fa2_used
    fa2_used = True
for module in model.modules():
    if isinstance(module, M2M100FlashAttention2):
        module.register_forward_hook(hook_fn)
input_text = 'This is a test sentence.'
traincheck.instrumentor.tracer.DISABLE_WRAPPER = True
input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
traincheck.instrumentor.tracer.DISABLE_WRAPPER = False
decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = torch.tensor([[decoder_start_token_id]]).to(device)
outputs = []
for _ in range(3):
    with torch.no_grad(), torch.cuda.amp.autocast():
        output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits
        outputs.append(output)
if fa2_used:
    print('Confirmed: FlashAttention2 was used during the forward pass.')
else:
    print('Warning: FlashAttention2 was not used in the forward pass.')
all_equal = all((torch.allclose(outputs[i], outputs[i + 1], atol=1e-05) for i in range(len(outputs) - 1)))
if all_equal:
    print('Bug not exposed: Outputs are consistent across runs in evaluation mode.')
else:
    print('Bug exposed: Outputs vary across runs in evaluation mode due to dropout not being disabled!')