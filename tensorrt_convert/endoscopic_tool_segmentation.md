##  Description

Cannot export a model to TensorRT after successfully transfered to torchscript.

## To Reproduce

Steps to reproduce the behavior:

1. Pull the monai image 1.1.0 from [link](https://hub.docker.com/r/projectmonai/monai/tags)
1. Start a docker container with the downloaded image in step 1.
1. Run the code below.

```
import torch
import torch_tensorrt
from monai.networks.nets import FlexibleUNet
import monai

if __name__ == "__main__":
    input_size = (1, 3, 480, 736)
    print(monai.__file__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlexibleUNet(
        in_channels=3, out_channels=2, backbone="efficientnet-b2", is_pad=False
    )

    model.to(device=device)
    model.eval()
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
DEBUG: [Torch-TensorRT] - Settings requested for Torch Fallback:
    "enabled": True
    "min_block_size": 3
    "torch_executed_operators": [
     ]
DEBUG: [Torch-TensorRT] - Parititioning source module into PyTorch and TensorRT sub blocks
DEBUG: [Torch-TensorRT] - In progress TRT block does not meet minimum block size requirements, therefore folding into in progress PyTorch block
DEBUG: [Torch-TensorRT] - Finalizing in progress Torch block
DEBUG: [Torch-TensorRT] - Segment Block @0:
    Target: Torch

    Graph: graph(%x.79 : Tensor):
  %3 : float[] = prim::Constant[value=[2., 2.]]()
  %self.encoder._conv_stem.bias.39 : NoneType = prim::Constant()
  %0 : Tensor = aten::upsample_nearest1d(%x.79, %self.encoder._conv_stem.bias.39, %3) # /opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:3916:15
  return ()


DEBUG: [Torch-TensorRT] - Registering input/output torch::jit::Value for segmented graphs
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
  File "/home/liubin/data/trt_bundle_experiment/export_flexible_unet_trt.py", line 32, in <module>
    trt_ts_model = torch_tensorrt.compile(
  File "/opt/conda/lib/python3.8/site-packages/torch_tensorrt/_compile.py", line 125, in compile
    return torch_tensorrt.ts.compile(
  File "/opt/conda/lib/python3.8/site-packages/torch_tensorrt/ts/_compiler.py", line 136, in compile
    compiled_cpp_mod = _C.compile_graph(module._c, _parse_compile_spec(spec))
RuntimeError: The following operation failed in the TorchScript interpreter.
Traceback of TorchScript (most recent call last):
  File "/opt/monai/monai/networks/nets/flexible_unet.py", line 337, in forward
        x = inputs
        enc_out = self.encoder(x)
        decoder_out = self.decoder(enc_out, self.skip_connect)
                      ~~~~~~~~~~~~ <--- HERE
        x_seg = self.segmentation_head(decoder_out)
    
  File "/opt/monai/monai/networks/nets/flexible_unet.py", line 166, in forward
            else:
                skip = None
            x = block(x, skip)
                ~~~~~ <--- HERE
    
        return x
  File "/opt/monai/monai/networks/nets/basic_unet.py", line 157, in forward
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)
              ~~~~~~~~~~~~~ <--- HERE
    
        if x_e is not None:
  File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/container.py", line 204, in forward
    def forward(self, input):
        for module in self:
            input = module(input)
                    ~~~~~~ <--- HERE
        return input
  File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/upsampling.py", line 156, in forward
    def forward(self, input: Tensor) -> Tensor:
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,
               ~~~~~~~~~~~~~ <--- HERE
                             recompute_scale_factor=self.recompute_scale_factor)
  File "/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py", line 3916, in interpolate

    if input.dim() == 3 and mode == "nearest":
        return torch._C._nn.upsample_nearest1d(input, output_size, scale_factors)
               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
    if input.dim() == 4 and mode == "nearest":
        return torch._C._nn.upsample_nearest2d(input, output_size, scale_factors)
RuntimeError: It is expected output_size equals to 1, but got size 2
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
 - Any other relevant information:bundle_version:0.3.2

## Additional context

<!-- Add any other context about the problem here. -->
I can transfer this model to a onnx model and then covert to a TensorRT engine by runing the command below.
```
trtexec --onnx=models/model.onnx --saveEngine=models/model.trt --fp16 --minShapes=INPUT__0:1x3x736x480 --optShapes=INPUT__0:4x3x736x480 --maxShapes=INPUT__0:8x3x736x480 --shapes=INPUT__0:4x3x736x480
```