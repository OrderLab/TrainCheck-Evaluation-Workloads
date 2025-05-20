import torch
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/home/yuxuan/TrainCheck-Evaluation-Workloads/silent-issue-detection/bug-reprod-scripts/x-jxmnop-ddp-out-of-sync/traincheck_run_bug_torch_2.2.2+cu121_2025-05-18_14-21-26"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
from traincheck.instrumentor import VarSampler
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from torch import nn, optim
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(nn, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(optim, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from torch.nn.parallel import DistributedDataParallel as DDP
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(DDP, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from transformers import Trainer, TrainingArguments, AutoModel, AutoTokenizer
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from traincheck import annotate_stage
from traincheck.instrumentor import meta_vars
annotate_stage('init')
torch.distributed.init_process_group(backend='nccl')
import os
meta_vars['_DATA_PARALLEL_RANK'] = os.environ['RANK']
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model_sampler = VarSampler(model, var_name='model')
dataset = load_dataset('glue', 'stsb')

def preprocess(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=128)
tokenized_datasets = dataset.map(preprocess, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

def get_model_ddp_or_not(step):
    if step % 2 == 0:
        return model
    else:
        return DDP(model)
training_args = TrainingArguments(output_dir='./results', per_device_train_batch_size=8, num_train_epochs=1, logging_steps=10, max_steps=10, local_rank=torch.distributed.get_rank())

def compute_contrastive_loss(model, inputs):
    outputs = model(input_ids=inputs['input_ids'].to(training_args.device), attention_mask=inputs['attention_mask'].to(training_args.device))
    embeddings = outputs.last_hidden_state.mean(dim=1)
    batch_size = embeddings.size(0)
    labels = torch.cat([torch.arange(batch_size // 2), torch.arange(batch_size // 2)]).to(training_args.device)
    contrastive_loss_fn = nn.CrossEntropyLoss()
    similarities = torch.matmul(embeddings, embeddings.T)
    contrastive_loss = contrastive_loss_fn(similarities, labels)
    return contrastive_loss

class BuggyTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=training_args.learning_rate)
        model_sampler.register_hook(self.optimizer)

    def training_step(self, model, inputs):
        annotate_stage('training')
        step = self.state.global_step
        meta_vars['step'] = step
        buggy_model = get_model_ddp_or_not(step)
        loss = compute_contrastive_loss(buggy_model, inputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f'Rank {torch.distributed.get_rank()} | Loss: {loss.item()}')
        return loss.detach()
trainer = BuggyTrainer(model=model, args=training_args, train_dataset=tokenized_datasets['train'])
annotate_stage('training')
trainer.train(resume_from_checkpoint=get_last_checkpoint('./results'))