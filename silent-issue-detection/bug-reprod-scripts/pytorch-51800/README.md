# PYTORCH-51800 Issue

https://github.com/pytorch/pytorch/issues/51800

## Environment Setup

Python 3.12 with Pytorch 2.3.1+cu121 will be enough to reproduce the bug

## Run

python3 bug_repro.py > result.txt
Check the result to see the missing initialization of weights

## What to expect

We should expect that before any evaluation or training (as soon as the model instance is created), the weight parameters of all layers are initialized properly as we want.

## What is the issue

The issue is that if we create an `nn.Linear` layer, denoted as `fc1` for example and if we use `nn.spactral_norm(fc1)` to normalize it. The parameters are initialized when the first `forward()` call is triggered. If one is trying to check the quaility of the model's parameter initialization or parameter's variance, he or she may set model to `eval()` mode directly. This will cause trouble since if the model is directly set to `eval()`, `nn.spactral_norm` will not be well initialized before the evaluation. So this evaluation cannot reflect the true 
quality of the model paramter initialization and can mislead the programmer to make some GAN training strategies (potential resourse waste here). 

## Root cause
First, in `SpectralNorm.apply` function, `SpectralNorm.__call__` method is registered to forward hook:  
```python
@staticmethod
def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float) -> 'SpectralNorm':
...
    fn = SpectralNorm(name, n_power_iterations, dim, eps)
...
    module.register_forward_pre_hook(fn)
    module._register_state_dict_hook(SpectralNormStateDictHook(fn))
    module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
    return fn
```

As the model propogates forward, `__call__` method in `torch.nn.Module` will be triggered:  
```python
def __call__(self, *args, **kwargs):
    return self._call_impl(*args, **kwargs)
```
Then, `_call_impl` method will forward propogate hooks:  
```python
def _call_impl(self, *args, **kwargs):
    ...
    if _global_forward_pre_hooks or self._forward_pre_hooks:
        for hook_id, hook in (
            *_global_forward_pre_hooks.items(),
            *self._forward_pre_hooks.items(),
        ):
            if hook_id in self._forward_pre_hooks_with_kwargs:
                args_kwargs_result = hook(self, args, kwargs)
                if args_kwargs_result is not None:
                    if isinstance(args_kwargs_result, tuple) and len(args_kwargs_result) == 2:
                        args, kwargs = args_kwargs_result
                    else:
                        raise RuntimeError(
                            "forward pre-hook must return None or a tuple "
                            f"of (new_args, new_kwargs), but got {args_kwargs_result}."
                        )
            else:
                args_result = hook(self, args)
                if args_result is not None:
                    if not isinstance(args_result, tuple):
                        args_result = (args_result,)
                    args = args_result
    ...
    result = forward_call(*args, **kwargs)
    ...
    return result
```
`hook` is the registered `SpectralNorm` instance, so `SpectralNorm`'s `__call__` method will be triggered:  
 ```python
 def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))
 ```
In `compute_weight`, we see that:  
 ```python
 def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
...
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma
        return weight
  ```
In training mode, `do_power_iteration` will be `true` and the weights of this layer will be updated. However, in evaluation mode `do_power_iteration` will be `False`, no update will happen on the parameters in this layer. So no initialization will happen unless forward is triggered in training mode.  

## How to Fix

<!-- This bug has already been fixed -->

Before the evluation, for user, do a dumy `forward()` operation in the training mode and then set the model to the evaluation mode. Then, everything is initialized and one can do evaluation properly.  
For example, add:  
```python
model = model.train()
z = model.forward(x)
y = model.eval()(x)
```

## Potential Ways to Detect the Bug Automatically

<!-- I think this bug is very hard to be captured with the invariant approach -->

Do Invariant check:  
   1. Imply:  
   If the model contains `nn.spectral_norm` and the model's `forward()` function is called in the evaluation mode with `do_power_iteration == False` at the first time. Raise a warning that the prediction may not be correct since no initialization happens.
       
### Example Code for Invariant Check

TBD
