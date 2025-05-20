import torch
from transformers import GPT2Model, GPT2Config
configuration = GPT2Config()
model = GPT2Model(configuration)