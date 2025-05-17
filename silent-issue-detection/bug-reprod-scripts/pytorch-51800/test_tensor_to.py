import torch

a = torch.randn(5,5)
print("instrumented: ", getattr(a.to, "_traincheck_instrumented", False))
print("trace dump disabled: ", getattr(a.to, "_traincheck_dump_disabled", False))
a.to("cuda")