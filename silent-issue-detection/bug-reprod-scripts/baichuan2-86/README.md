**Original Issue Post:**
https://github.com/baichuan-inc/Baichuan2/issues/86#issuecomment-1728458018

**Manifest:**
1. Slow Inference Speed
2. Bad Output Quality
3. Indeterminstic Exceptions caused by Infs and NaNs

> @shesung: "我试了一下，可以加载到两张卡上。但是，推理的速度非常慢，大概只有单卡的20%速度。并且很容易胡言乱语，或者出现inf,nan的报错。" | "I tried it and was able to load it onto two cards. However, the inference speed is very slow, only about 20% of the speed of a single card. Additionally, it often generates nonsense or errors like inf or nan."

> @AnitaSherry: "一样，单卡没问题，多卡必报错" | "Same here. No issues on a single card, but errors occur every time on multiple cards."

**Root Cause:**
1. Bad GPU P2P communication setup leading to dataloss during data transmission.

**Reproduction Script:**
This is a hardware related one and needs fault injection for reproduction.
For simplicity, all the issues related to data loss will be reproduced by [pytorch-84803](../pytorch-84803).


