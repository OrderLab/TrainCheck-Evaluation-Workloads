import torch
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/home/yuxuan/TrainCheck-Evaluation-Workloads/silent-issue-detection/bug-reprod-scripts/transformers-17877/traincheck_run_bug_torch_2.2.2+cu121_transformers_4.20.1_2025-05-19_17-28-31"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = True
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from transformers import GPT2Model, GPT2Config
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(GPT2Model, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(GPT2Config, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
configuration = GPT2Config()
model = GPT2Model(configuration)