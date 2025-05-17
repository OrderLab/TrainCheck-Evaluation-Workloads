Original Issue: https://github.com/huggingface/transformers/pull/17877

**Manifest:**
Initialization methods being repeatedly invoked on model (GPT2)'s parameters, causing potentially slow initialization.

To reproduce, install 
```bash
transformers==4.20.1
```