# -*- coding: utf-8 -*-
# the above line is for the `prompt_for_error`

from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="English", task="transcribe")

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

it = iter(load_dataset("librispeech_asr", "all", split="test.other", streaming=True))
while it:
  _ = [next(it) for x in range(3)]
  clip = next(it)
  if clip["id"] == '7902-96592-0026':
    break

input_features = processor(clip['audio']['array'], sampling_rate=clip['audio']['sampling_rate'], return_tensors="pt").input_features


# Example of it not limiting generation to max_new_tokens when prompt_ids length too large 
long_prompt = 5 * "Bubalina is a subtribe of wild cattle that includes the various species of true buffalo. Species include the African buffalo, the anoas, and the wild water buffalo (including the domesticated variant water buffalo. Buffaloes can be found naturally in sub-Saharan Africa, South Asia and Southeast Asia, and domestic and feral populations have been introduced to Europe, the Americas, and Australia. In addition to the living species, bubalinans have an extensive fossil record where remains have been found in much of Afro-Eurasia."
prompt_ids = processor.get_prompt_ids(long_prompt)
pred_ids = model.generate(input_features, language="english", task="transcribe", max_new_tokens=10, prompt_ids=prompt_ids)
decoded = processor.decode(pred_ids[0], skip_special_tokens=True)
new_tokens = processor.tokenizer(decoded, add_special_tokens=False)["input_ids"]
print(new_tokens)
print(len(new_tokens))

# Example of erroring
prompt_for_error = "some text rich in domain specific vocabulary lives here - I wish you would believe me that I am in as great trouble about it as you are - then as archiestered in the dark literally a gas for the astonishment here at the faint and wrestling once more and again all with silent - I'll soon show them that I am not going to be played with - to do this he must scheme lie head till morning then make for the nearest point it's signal for help I also boats crew were already searching for him how to escape - no that was too bad you cannot do that - but there was no chance for his body there the head would not go first - shall I come to father? no - what a queer dream he thought to himself - and I am hungry too 今晚會是我 再回家吧 - oh those bars he meant 雷 exclaimed and he was  advancing towards them, and just as he drew near there was a wrestling noise nd to the window a couple of hands seized the bars there was a scratching of 布側 against stonework and ram スペース 敬射的 金融 敬射的 金融 敬射的 金融 敬射的 金融 敬射的 金融 敬射的 金融 � - I saw you last night and wondered whose boy he was - I think I don't know you Mr. Orphazard "
prompt_ids = processor.get_prompt_ids(prompt_for_error)
pred_ids = model.generate(input_features, language="english", task="transcribe", max_new_tokens=128, prompt_ids=prompt_ids)
decoded = processor.decode(pred_ids[0], skip_special_tokens=True)
new_tokens = processor.tokenizer(decoded, add_special_tokens=False)["input_ids"]

print(new_tokens)
print(len(new_tokens))
assert len(new_tokens) <= 128, f"Expected length <= 128, got {len(pred_ids[0])}"