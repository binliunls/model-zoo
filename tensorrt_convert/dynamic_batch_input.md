##  Description

Cannot export a model to TensorRT when tring to config the input as dynamic batch input. This can be fixed by make the `min_input_shape`, `opt_input_shape` and `max_input_shape` of the `inputs` parameter in the `torch_tensorrt.compile` to the same size. The original input_shape is `(1, 1, 96, 96, 96)` in this program. 

Related issues: (1577)(https://github.com/pytorch/TensorRT/discussions/1577)

## To Reproduce

Steps to reproduce the behavior:

1. Pull the monai image 1.1.0 from [link](https://hub.docker.com/r/projectmonai/monai/tags)
1. Start a docker container with the downloaded image in step 1.
1. Run the code below.

```
import os
import ast
import sys
import glob
import torch
import monai
import monai.bundle
from monai.bundle import ConfigParser
from typing import Union, Sequence, Tuple
import torch_tensorrt


def get_input_shape(parser):

    # define inner tool function to parse input shape
    def _get_var_names(expr: str):
        """
        Parse the expression and discover what variables are present in it based on ast module.

        Args:
            expr: source expression to parse.

        """
        tree = ast.parse(expr)
        return [m.id for m in ast.walk(tree) if isinstance(m, ast.Name)]

    def _get_fake_spatial_shape(
        shape: Sequence[Union[str, int]], p: int = 1, n: int = 1, any: int = 1
    ) -> Tuple:
        """
        Get spatial shape for fake data according to the specified shape pattern.
        It supports `int` number and `string` with formats like: "32", "32 * n", "32 ** p", "32 ** p *n".

        Args:
            shape: specified pattern for the spatial shape.
            p: power factor to generate fake data shape if dim of expected shape is "x**p", default to 1.
            p: multiply factor to generate fake data shape if dim of expected shape is "x*n", default to 1.
            any: specified size to generate fake data shape if dim of expected shape is "*", default to 1.

        """
        ret = []
        for i in shape:
            if isinstance(i, int):
                ret.append(i)
            elif isinstance(i, str):
                if i == "*":
                    ret.append(any)
                else:
                    for c in _get_var_names(i):
                        if c not in ["p", "n"]:
                            raise ValueError(
                                f"only support variables 'p' and 'n' so far, but got: {c}."
                            )
                    ret.append(eval(i, {"p": p, "n": n}))
            else:
                raise ValueError(
                    f"spatial shape items must be int or string, but got: {type(i)} {i}."
                )
        return tuple(ret)

    if "_meta_#network_data_format#inputs#latent" in parser:
        input_channels = parser["_meta_#network_data_format#inputs#latent#num_channels"]
        input_spatial_shape = parser[
            "_meta_#network_data_format#inputs#latent#spatial_shape"
        ]
    else:
        input_channels = parser["_meta_#network_data_format#inputs#image#num_channels"]
        input_spatial_shape = parser[
            "_meta_#network_data_format#inputs#image#spatial_shape"
        ]
    spatial_shape = _get_fake_spatial_shape(input_spatial_shape)
    if not input_channels:
        spatial_shape = (1, *spatial_shape)
    else:
        spatial_shape = (1, input_channels, *spatial_shape)
    return spatial_shape


def get_model_instance(bundle, parser):

    bundle_model_path = os.path.join(bundle, "models", "model.pt")

    if "gnetwork" in parser:
        net_id = "gnetwork"
    elif "net" in parser:
        net_id = "net"
    elif "network_def" in parser:
        net_id = "network_def"
    else:
        net_id = "network"

    model = parser.get_parsed_content(net_id)
    if os.path.exists(bundle_model_path):
        model.load_state_dict(torch.load(bundle_model_path))
    return model


def get_config_parser(bundle):
    bundle_config_path = os.path.join(bundle, "configs")
    meta_json = os.path.join(bundle_config_path, "metadata.json")
    train_files = glob.glob(os.path.join(bundle_config_path, "train.*"))
    train_json = train_files[0]
    infer_files = glob.glob(os.path.join(bundle_config_path, "inference.*"))
    infer_json = infer_files[0]
    bundle_json = infer_json if os.path.exists(infer_json) else train_json
    parser = ConfigParser()
    parser.read_meta(meta_json)
    parser.read_config(bundle_json)
    return parser


def get_model_and_input_shape(bundle_path, bundle_name):
    """
    Get the pretrained model if model weight exists in bundle or random initialized model
    and input shape.
    """
    cur_bundle_root = os.path.join(bundle_path, bundle_name)
    sys.path.append(cur_bundle_root)
    config_parser = get_config_parser(cur_bundle_root)
    model = get_model_instance(cur_bundle_root, config_parser)
    input_shape = get_input_shape(config_parser)
    sys.path.remove(cur_bundle_root)
    return model, input_shape


def download_given_bundle(save_path, bundle_name):
    """
    Download a bundle named as bundle_name to save_path.
    """
    os.makedirs(save_path, exist_ok=True)
    dst_bundle_path = os.path.join(save_path, bundle_name)
    if os.path.exists(dst_bundle_path):
        print(f"{bundle_name} exists in '{save_path}', skiping download")
        return
    else:
        bundle_tuple = monai.bundle.get_all_bundles_list()
        bundle_dir = save_path
        bundle_list, version_list = list(zip(*bundle_tuple))
        bundle_index = bundle_list.index(bundle_name)
        bundle_version = version_list[bundle_index]
        monai.bundle.download(
            name=bundle_name,
            version=bundle_version,
            bundle_dir=bundle_dir,
        )
        print(f"Successfully downloaded {bundle_name}, version {bundle_version}.")


def trt_convert(model, input_shape):
    """
    Converting the input model to tensorRT-engine script in pytorch.
    """
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        jit_model = torch.jit.script(model)

    min_input_shape = [*input_shape]
    min_input_shape[0] *= 1
    opt_input_shape = [*input_shape]
    opt_input_shape[0] *= 4
    max_input_shape = [*input_shape]
    max_input_shape[0] *= 32

    inputs = [
        torch_tensorrt.Input(
            min_shape=min_input_shape,
            opt_shape=opt_input_shape,
            max_shape=max_input_shape,
            dtype=torch.float,
        )
    ]
    enabled_precision = {torch.float, torch.half}
    with torch_tensorrt.logging.graphs():
        trt_ts_model = torch_tensorrt.compile(
            jit_model, inputs=inputs, enabled_precisions=enabled_precision
        )
    # torch.jit.save(trt_ts_model, out_name)
    return trt_ts_model


def convert_to_trt(save_path, bundle_name):
    # download a bundle to save path
    download_given_bundle(save_path, bundle_name)

    # get model with pretrained weight if exists and input shape
    model, input_shape = get_model_and_input_shape(save_path, bundle_name)

    # convert model to trt engine through torch-trt
    trt_model = trt_convert(model, input_shape)

    return trt_model


if __name__ == "__main__":
    SAVE_PATH = r"/workspace/bundle"
    BUNDLE_NAME = "spleen_ct_segmentation"
    trt_model = convert_to_trt(SAVE_PATH, BUNDLE_NAME)


```

Output was:

```
DEBUG: [Torch-TensorRT] - In MapInputsAndDetermineDTypes, the g->inputs() size is 1, CollectionInputSpecMap size is1
INFO: [Torch-TensorRT] - Method requested cannot be compiled end to end by Torch-TensorRT.TorchScript.
Unsupported operators listed below:
  - aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor

DEBUG: [Torch-TensorRT] - Unsupported operator: aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor
  File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/conv.py", line 1108
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]
    
        return F.conv_transpose3d(
               ~~~~~~~~~~~~~~~~~~ <--- HERE
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

Traceback (most recent call last):
  File "/opt/conda/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/opt/conda/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/root/.vscode-server/extensions/ms-python.python-2023.2.0/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher/../../debugpy/__main__.py", line 39, in <module>
    cli.main()
  File "/root/.vscode-server/extensions/ms-python.python-2023.2.0/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 430, in main
    run()
  File "/root/.vscode-server/extensions/ms-python.python-2023.2.0/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 284, in run_file
    runpy.run_path(target, run_name="__main__")
  File "/root/.vscode-server/extensions/ms-python.python-2023.2.0/pythonFiles/lib/python/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 321, in run_path
    return _run_module_code(code, init_globals, run_name,
  File "/root/.vscode-server/extensions/ms-python.python-2023.2.0/pythonFiles/lib/python/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 135, in _run_module_code
    _run_code(code, mod_globals, init_globals,
  File "/root/.vscode-server/extensions/ms-python.python-2023.2.0/pythonFiles/lib/python/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 124, in _run_code
    exec(code, run_globals)
  File "/home/liubin/data/trt_bundle_experiment/convert_bundle_trt.py", line 200, in <module>
    trt_model = convert_to_trt(SAVE_PATH, BUNDLE_NAME)
  File "/home/liubin/data/trt_bundle_experiment/convert_bundle_trt.py", line 192, in convert_to_trt
    trt_model = trt_convert(model, input_shape)
  File "/home/liubin/data/trt_bundle_experiment/convert_bundle_trt.py", line 177, in trt_convert
    trt_ts_model = torch_tensorrt.compile(
  File "/opt/conda/lib/python3.8/site-packages/torch_tensorrt/_compile.py", line 125, in compile
    return torch_tensorrt.ts.compile(
  File "/opt/conda/lib/python3.8/site-packages/torch_tensorrt/ts/_compiler.py", line 136, in compile
    compiled_cpp_mod = _C.compile_graph(module._c, _parse_compile_spec(spec))
RuntimeError: Trying to create tensor with negative dimension -1: [-1, 1, 96, 96, 96]
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
 - Any other relevant information:bundle_version:

## Additional context

<!-- Add any other context about the problem here. -->
