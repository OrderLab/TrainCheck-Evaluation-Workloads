# import torch

# import inspect

# import logging
# import sys
# import threading
# import time
# import traceback
# import uuid
# from importlib.machinery import ModuleSpec
# from typing import Any

# import pandas as pd
# import torch

# THREAD_LOCAL = threading.local()


# def safe_getattr(obj, attr, default=None):
#     """Safely get the attribute of an object.
#     try except is necessary as some objects (e.g. cuBLASModule in PyTorch) might have custom __getattr__
#     method that raises an exception when accessing certain attributes.
#     """
#     try:
#         return getattr(obj, attr, default)
#     except Exception as e:
#         if isinstance(e, AssertionError):
#             return default
#         if isinstance(e, RuntimeError):
#             if (
#                 str(e)
#                 in "RuntimeError: Tried to instantiate class '__qualname__.__qualname__', but it does not exist! Ensure that it is registered via torch::class_"
#             ):
#                 return default
#         if isinstance(e, ModuleNotFoundError):
#             return default
#         raise


# def typename(o):
#     if isinstance(o, torch.nn.Parameter):
#         return "torch.nn.Parameter"
#     if isinstance(o, torch.Tensor):
#         return o.type()
#     module = safe_getattr(o, "__module__", "")
#     if isinstance(module, ModuleSpec):
#         # handle the case when module is a ModuleSpec object
#         module = module.name
#     if module in ["buitins", "__builtin__", None]:
#         module = ""
#     class_name = safe_getattr(o, "__qualname__", "")
#     if not isinstance(
#         class_name, str
#     ):  # the instance here is for the case when __qualname__ is _ClassNamespace
#         class_name = ""
#     if not class_name:
#         class_name = safe_getattr(o, "__name__", "")
#     if not class_name:
#         class_name = safe_getattr(o, "__class__", type(o)).__name__
#     assert isinstance(module, str) and isinstance(
#         class_name, str
#     ), f"module and class_name should be str, but got {module} and {class_name} for {o}"
#     return f"{module}.{class_name}" if module else class_name


# print(inspect.isbuiltin(torch.Tensor.to_dense))
# print(typename(torch.Tensor.to_dense))
# print(dir(torch.TensorBase))


pattern = "Reason: Skipping attribute as it is None, Module: "
unique_items = set()
with open("trace_API_3499981_140254285555520.log", "r") as f:
    for line in f:
        if pattern in line:
            unique_items.add(line.split(pattern)[-1].strip())
print(f"Unique items: {len(unique_items)}")
print(unique_items)

