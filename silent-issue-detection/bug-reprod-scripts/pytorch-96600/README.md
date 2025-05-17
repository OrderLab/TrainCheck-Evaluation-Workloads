**Original Issue Post:**
https://github.com/pytorch/pytorch/issues/96600

**Manifest:**
1. Dataloss when doing Data Parallel training on setups with multiple 4090 cards.

**Root Cause:**
1. Bad GPU P2P communication setup leading to dataloss during data transmission.

**Reproduction Script:**
This is a hardware related one and needs fault injection for reproduction.
For simplicity, all the issues related to data loss will be reproduced by [pytorch-84803](../pytorch-84803).


