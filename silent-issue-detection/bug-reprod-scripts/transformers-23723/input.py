# -*- coding: utf-8 -*-

from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

it = iter(load_dataset("librispeech_asr", "all", split="test.other", streaming=True, trust_remote_code=True))
while it:
  _ = [next(it) for x in range(3)]
  clip = next(it)
  if clip["id"] == '7902-96592-0026':
    break
# release it

import traincheck

traincheck.annotate_answer_start_token_ids(processor.tokenizer.additional_special_tokens_ids[1], include_start_token=True)

input_features = processor(clip['audio']['array'], sampling_rate=clip['audio']['sampling_rate'], return_tensors="pt").input_features
# Example of it not limiting generation to max_new_tokens when prompt_ids length too large 
long_prompt = "TrainCheck is a great tool for making training correct with low-overhead"
prompt_ids = processor.get_prompt_ids(long_prompt)
pred_ids = model.generate(input_features, language="english", task="transcribe", max_new_tokens=128, prompt_ids=prompt_ids)
print(processor.decode(pred_ids[0]))
print(pred_ids)


# Example of it not limiting generation to max_new_tokens when prompt_ids length too large 
long_prompt = "TrainCheck is a great tool for making training correct with low-overhead and it doesn't introduce any tricky changes to the existing codebase"
prompt_ids = processor.get_prompt_ids(long_prompt)
pred_ids = model.generate(input_features, language="english", task="transcribe", max_new_tokens=128, prompt_ids=prompt_ids)

# Example of it not limiting generation to max_new_tokens when prompt_ids length too large 
long_prompt = "TrainCheck is a great tool for making training correct with low-overhead and it doesn't introduce any tricky changes to the existing codebase, everything is so automated"
prompt_ids = processor.get_prompt_ids(long_prompt)
pred_ids = model.generate(input_features, language="english", task="transcribe", max_new_tokens=128, prompt_ids=prompt_ids)

# Example of it not limiting generation to max_new_tokens when prompt_ids length too large 
long_prompt = "TrainCheck is a great tool for making training correct with low-overhead and it doesn't introduce any tricky changes to the existing codebase, everything is so automated and I am sure it will be a very practical tool for the developers"
prompt_ids = processor.get_prompt_ids(long_prompt)
pred_ids = model.generate(input_features, language="english", task="transcribe", max_new_tokens=128, prompt_ids=prompt_ids)