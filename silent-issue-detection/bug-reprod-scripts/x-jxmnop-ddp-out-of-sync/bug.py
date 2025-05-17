import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Trainer, TrainingArguments, AutoModel, AutoTokenizer
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

from traincheck import annotate_stage
from traincheck.instrumentor import meta_vars

annotate_stage("init")

# Initialize distributed training
torch.distributed.init_process_group(backend="nccl")

import os
meta_vars["_DATA_PARALLEL_RANK"] = os.environ['RANK']

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load STS-B dataset
dataset = load_dataset("glue", "stsb")
def preprocess(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=128)
tokenized_datasets = dataset.map(preprocess, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Wrap model in DDP conditionally to simulate bug
def get_model_ddp_or_not(step):
    if step % 2 == 0:  # sometimes nn.Module, sometimes DDP
        return model
    else:
        return DDP(model)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_steps=10,
    local_rank=torch.distributed.get_rank()  # for DDP
)

# Contrastive loss
def compute_contrastive_loss(model, inputs):
    outputs = model(input_ids=inputs["input_ids"].to(training_args.device), attention_mask=inputs["attention_mask"].to(training_args.device))
    embeddings = outputs.last_hidden_state.mean(dim=1)
    batch_size = embeddings.size(0)
    labels = torch.cat([torch.arange(batch_size // 2), torch.arange(batch_size // 2)]).to(training_args.device)
    contrastive_loss_fn = nn.CrossEntropyLoss()
    similarities = torch.matmul(embeddings, embeddings.T)
    contrastive_loss = contrastive_loss_fn(similarities, labels)
    return contrastive_loss

# Trainer with the intentional bug and optimizer
class BuggyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=training_args.learning_rate)  # Define optimizer
    
    def training_step(self, model, inputs):
        step = self.state.global_step
        meta_vars["step"] = step
        buggy_model = get_model_ddp_or_not(step)
        loss = compute_contrastive_loss(buggy_model, inputs)
        
        self.optimizer.zero_grad()  # Reset gradients
        loss.backward()
        self.optimizer.step()  # Update model parameters
        
        # Print loss on each GPU to observe desync
        print(f"Rank {torch.distributed.get_rank()} | Loss: {loss.item()}")
        return loss.detach()

# Train
trainer = BuggyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
)
annotate_stage("training")
trainer.train(resume_from_checkpoint=get_last_checkpoint("./results"))
