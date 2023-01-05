##  Description

Cannot export to torchscript.

## To Reproduce

Steps to reproduce the behavior:

1. Pull the monai image 1.1.0 from [link](https://hub.docker.com/r/projectmonai/monai/tags)
1. Star a docker container with the downloaded image in step 1.
1. Run the code below.

```
import torch
import torch_tensorrt
from monai.networks.nets import SwinUNETR
import monai

if __name__ == "__main__":
    input_size = (1, 1, 96, 96, 96)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SwinUNETR(
        spatial_dims=3,
        img_size=96,
        in_channels=1,
        out_channels=14,
        feature_size=48,
        use_checkpoint=True,
    )
    model.to(device=device)

    model.eval()
    with torch.no_grad():
        ts_model = torch.jit.script(model)

    inputs = [
        torch_tensorrt.Input(
            min_shape=input_size,
            opt_shape=input_size,
            max_shape=input_size,
        )
    ]
    enabled_precision = {torch.float, torch.half}
    with torch_tensorrt.logging.debug():
        trt_ts_model = torch_tensorrt.compile(
            ts_model, inputs=inputs, enabled_precisions=enabled_precision
        )
    torch.jit.save(trt_ts_model, "model_trt.ts")

```

Output was:

```
Traceback (most recent call last):
  File "/opt/conda/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/opt/conda/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/root/.vscode-server/extensions/ms-python.python-2022.20.1/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher/../../debugpy/__main__.py", line 39, in <module>
    cli.main()
  File "/root/.vscode-server/extensions/ms-python.python-2022.20.1/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 430, in main
    run()
  File "/root/.vscode-server/extensions/ms-python.python-2022.20.1/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 284, in run_file
    runpy.run_path(target, run_name="__main__")
  File "/root/.vscode-server/extensions/ms-python.python-2022.20.1/pythonFiles/lib/python/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 321, in run_path
    return _run_module_code(code, init_globals, run_name,
  File "/root/.vscode-server/extensions/ms-python.python-2022.20.1/pythonFiles/lib/python/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 135, in _run_module_code
    _run_code(code, mod_globals, init_globals,
  File "/root/.vscode-server/extensions/ms-python.python-2022.20.1/pythonFiles/lib/python/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 124, in _run_code
    exec(code, run_globals)
  File "/home/liubin/data/trt_bundle_experiment/export_flexible_unet_trt.py", line 59, in <module>
    ts_model = torch.jit.script(model)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_script.py", line 1286, in script
    return torch.jit._recursive.create_script_module(
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 476, in create_script_module
    return create_script_module_impl(nn_module, concrete_type, stubs_fn)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 538, in create_script_module_impl
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_script.py", line 615, in _construct
    init_fn(script_module)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 516, in init_fn
    scripted = create_script_module_impl(orig_value, sub_concrete_type, stubs_fn)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 538, in create_script_module_impl
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_script.py", line 615, in _construct
    init_fn(script_module)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 516, in init_fn
    scripted = create_script_module_impl(orig_value, sub_concrete_type, stubs_fn)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 538, in create_script_module_impl
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_script.py", line 615, in _construct
    init_fn(script_module)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 516, in init_fn
    scripted = create_script_module_impl(orig_value, sub_concrete_type, stubs_fn)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 538, in create_script_module_impl
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_script.py", line 615, in _construct
    init_fn(script_module)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 516, in init_fn
    scripted = create_script_module_impl(orig_value, sub_concrete_type, stubs_fn)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 538, in create_script_module_impl
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_script.py", line 615, in _construct
    init_fn(script_module)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 516, in init_fn
    scripted = create_script_module_impl(orig_value, sub_concrete_type, stubs_fn)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 542, in create_script_module_impl
    create_methods_and_properties_from_stubs(concrete_type, method_stubs, property_stubs)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 393, in create_methods_and_properties_from_stubs
    concrete_type._create_methods_and_properties(property_defs, property_rcbs, method_defs, method_rcbs, method_defaults)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_recursive.py", line 863, in try_compile_fn
    return torch.jit.script(fn, _rcb=rcb)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/_script.py", line 1340, in script
    ast = get_jit_def(obj, obj.__name__)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/frontend.py", line 293, in get_jit_def
    return build_def(parsed_def.ctx, fn_def, type_line, def_name, self_name=self_name, pdt_arg_types=pdt_arg_types)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/frontend.py", line 331, in build_def
    param_list = build_param_list(ctx, py_def.args, self_name, pdt_arg_types)
  File "/opt/conda/lib/python3.8/site-packages/torch/jit/frontend.py", line 355, in build_param_list
    raise NotSupportedError(ctx_range, _vararg_kwarg_err)
torch.jit.frontend.NotSupportedError: Compiled functions can't take variable number of arguments or use keyword-only arguments with defaults:
  File "/opt/conda/lib/python3.8/site-packages/torch/utils/checkpoint.py", line 164
def checkpoint(function, *args, use_reentrant: bool = True, **kwargs):
                                                             ~~~~~~~ <--- HERE
    r"""Checkpoint a model or part of the model
```
<!-- If you have a code sample, error messages, stack traces, please provide it here as well -->

## Expected behavior

Sucessfully convert the model.

## Environment

> Build information about Torch-TensorRT can be found by turning on debug messages
 - TensorRT: 8.5.0.12-1+cuda11.8
 - MONAI: 1.1.0
 - Torch-TensorRT Version (e.g. 1.0.0): 1.3.0a0
 - PyTorch Version (e.g. 1.0): 1.13.0a0
 - CPU Architecture: x86-64
 - OS (e.g., Linux): ubuntu 20.04
 - How you installed PyTorch (`conda`, `pip`, `libtorch`, source): conda
 - Build command you used (if compiling from source):
 - Are you using local sources or building from archives:
 - Python version:3.8.13
 - CUDA version: 11.8
 - GPU models and configuration: 3090Ti
 - Any other relevant information:bundle_version:0.3.9

## Additional context

<!-- Add any other context about the problem here. -->
I can transfer this model to a onnx model and then covert to a TensorRT engine by runing the command below.
```
trtexec --onnx=models/model.onnx --saveEngine=models/model.trt --fp16 --minShapes=INPUT__0:1x3x736x480 --optShapes=INPUT__0:4x3x736x480 --maxShapes=INPUT__0:8x3x736x480 --shapes=INPUT__0:4x3x736x480
```