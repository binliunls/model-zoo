##  Description

Cannot export a model to TensorRT after successfully transfered to torchscript.

## To Reproduce

Steps to reproduce the behavior:

1. Pull the monai image 1.1.0 from [link](https://hub.docker.com/r/projectmonai/monai/tags)
1. Start a docker container with the downloaded image in step 1.
1. Modify the `bundle_root` parameter in `train.yaml` or `inference.yaml`.
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

    inputs = [
        torch_tensorrt.Input(
            min_shape=input_shape,
            opt_shape=input_shape,
            max_shape=input_shape,
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
    BUNDLE_NAME = "pancreas_ct_dints_segmentation"
    trt_model = convert_to_trt(SAVE_PATH, BUNDLE_NAME)

```

Output was:

```
DEBUG: [Torch-TensorRT] - init_compile_spec with input vector
DEBUG: [Torch-TensorRT] - Settings requested for Lowering:
    torch_executed_modules: [
    ]
GRAPH: [Torch-TensorRT] - Before lowering: graph(%self : __torch__.monai.networks.nets.dints.DiNTS,
      %x.1 : Tensor):
  %57 : int = prim::Constant[value=-1]() # /opt/monai/monai/networks/nets/dints.py:507:50
  %start.1 : bool = prim::Constant[value=0]() # /opt/monai/monai/networks/nets/dints.py:505:16
  %33 : NoneType = prim::Constant()
  %7 : bool = prim::Constant[value=1]() # /opt/monai/monai/networks/nets/dints.py:493:8
  %18 : int = prim::Constant[value=0]() # /opt/monai/monai/networks/nets/dints.py:497:27
  %44 : int = prim::Constant[value=1]() # /opt/monai/monai/networks/nets/dints.py:504:36
  %154 : Tensor[] = prim::ListConstruct()
  %num_depths.1 : int = prim::GetAttr[name="num_depths"](%self)
   = prim::Loop(%num_depths.1, %7) # /opt/monai/monai/networks/nets/dints.py:493:8
    block0(%d.1 : int):
      %stem_down : __torch__.torch.nn.modules.container.___torch_mangle_135.ModuleDict = prim::GetAttr[name="stem_down"](%self)
      %12 : str = aten::str(%d.1) # /opt/monai/monai/networks/nets/dints.py:495:51
      %_mod_w.1 : InterfaceType<StemInterface> = prim::ModuleContainerIndex(%stem_down, %12)
      %x_out.1 : Tensor = prim::CallMethod[name="forward"](%_mod_w.1, %x.1) # /opt/monai/monai/networks/nets/dints.py:496:20
      %node_a.1 : Tensor = prim::GetAttr[name="node_a"](%self)
      %20 : Tensor = aten::select(%node_a.1, %18, %18) # /opt/monai/monai/networks/nets/dints.py:497:15
      %25 : Tensor = aten::select(%20, %18, %d.1) # /opt/monai/monai/networks/nets/dints.py:497:15
      %27 : bool = aten::Bool(%25) # /opt/monai/monai/networks/nets/dints.py:497:15
       = prim::If(%27) # /opt/monai/monai/networks/nets/dints.py:497:12
        block0():
          %30 : Tensor[] = aten::append(%154, %x_out.1) # /opt/monai/monai/networks/nets/dints.py:498:16
          -> ()
        block1():
          %38 : Tensor = aten::zeros_like(%x_out.1, %33, %33, %33, %33, %33) # /opt/monai/monai/networks/nets/dints.py:500:30
          %39 : Tensor[] = aten::append(%154, %38) # /opt/monai/monai/networks/nets/dints.py:500:16
          -> ()
      -> (%7)
  %dints_space : __torch__.monai.networks.nets.dints.TopologyInstance = prim::GetAttr[name="dints_space"](%self)
  %outputs.1 : Tensor[] = prim::CallMethod[name="forward"](%dints_space, %154) # /opt/monai/monai/networks/nets/dints.py:502:18
  %num_blocks : int = prim::GetAttr[name="num_blocks"](%self)
  %blk_idx.1 : int = aten::sub(%num_blocks, %44) # /opt/monai/monai/networks/nets/dints.py:504:18
  %47 : int[] = prim::ListConstruct(%18)
  %_temp.1 : Tensor = aten::empty(%47, %33, %33, %33, %33, %33) # /opt/monai/monai/networks/nets/dints.py:506:30
  %num_depths : int = prim::GetAttr[name="num_depths"](%self)
  %55 : int = aten::sub(%num_depths, %44) # /opt/monai/monai/networks/nets/dints.py:507:29
  %60 : int = aten::__range_length(%55, %57, %57) # /opt/monai/monai/networks/nets/dints.py:507:8
  %_temp : Tensor, %start : bool = prim::Loop(%60, %7, %_temp.1, %start.1) # /opt/monai/monai/networks/nets/dints.py:507:8
    block0(%62 : int, %_temp.37 : Tensor, %start.29 : bool):
      %res_idx.1 : int = aten::__derive_index(%62, %55, %57) # /opt/monai/monai/networks/nets/dints.py:507:8
      %stem_up : __torch__.torch.nn.modules.container.___torch_mangle_148.ModuleDict = prim::GetAttr[name="stem_up"](%self)
      %67 : str = aten::str(%res_idx.1) # /opt/monai/monai/networks/nets/dints.py:508:50
      %_mod_up.1 : InterfaceType<StemInterface> = prim::ModuleContainerIndex(%stem_up, %67)
      %_temp.35 : Tensor, %start.27 : bool = prim::If(%start.29) # /opt/monai/monai/networks/nets/dints.py:509:12
        block0():
          %74 : Tensor = aten::__getitem__(%outputs.1, %res_idx.1) # /opt/monai/monai/networks/nets/dints.py:510:40
          %77 : Tensor = aten::add(%74, %_temp.37, %44) # /opt/monai/monai/networks/nets/dints.py:510:40
          %_temp.5 : Tensor = prim::CallMethod[name="forward"](%_mod_up.1, %77) # /opt/monai/monai/networks/nets/dints.py:510:24
          -> (%_temp.5, %start.29)
        block1():
          %node_a : Tensor = prim::GetAttr[name="node_a"](%self)
          %86 : int = aten::add(%blk_idx.1, %44) # /opt/monai/monai/networks/nets/dints.py:511:29
          %88 : Tensor = aten::select(%node_a, %18, %86) # /opt/monai/monai/networks/nets/dints.py:511:17
          %93 : Tensor = aten::select(%88, %18, %res_idx.1) # /opt/monai/monai/networks/nets/dints.py:511:17
          %95 : bool = aten::Bool(%93) # /opt/monai/monai/networks/nets/dints.py:511:17
          %_temp.33 : Tensor, %start.25 : bool = prim::If(%95) # /opt/monai/monai/networks/nets/dints.py:511:12
            block0():
              %101 : Tensor = aten::__getitem__(%outputs.1, %res_idx.1) # /opt/monai/monai/networks/nets/dints.py:513:40
              %_temp.9 : Tensor = prim::CallMethod[name="forward"](%_mod_up.1, %101) # /opt/monai/monai/networks/nets/dints.py:513:24
              -> (%_temp.9, %7)
            block1():
              -> (%_temp.37, %start.29)
          -> (%_temp.33, %start.25)
      -> (%7, %_temp.35, %start.27)
  %stem_finals : __torch__.torch.nn.modules.container.___torch_mangle_151.Sequential = prim::GetAttr[name="stem_finals"](%self)
  %prediction.1 : Tensor = prim::CallMethod[name="forward"](%stem_finals, %_temp) # /opt/monai/monai/networks/nets/dints.py:514:21
  return (%prediction.1)

Traceback (most recent call last):
  File "/opt/conda/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/opt/conda/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/root/.vscode-server/extensions/ms-python.python-2022.20.2/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher/../../debugpy/__main__.py", line 39, in <module>
    cli.main()
  File "/root/.vscode-server/extensions/ms-python.python-2022.20.2/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 430, in main
    run()
  File "/root/.vscode-server/extensions/ms-python.python-2022.20.2/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 284, in run_file
    runpy.run_path(target, run_name="__main__")
  File "/root/.vscode-server/extensions/ms-python.python-2022.20.2/pythonFiles/lib/python/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 321, in run_path
    return _run_module_code(code, init_globals, run_name,
  File "/root/.vscode-server/extensions/ms-python.python-2022.20.2/pythonFiles/lib/python/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 135, in _run_module_code
    _run_code(code, mod_globals, init_globals,
  File "/root/.vscode-server/extensions/ms-python.python-2022.20.2/pythonFiles/lib/python/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 124, in _run_code
    exec(code, run_globals)
  File "/home/liubin/data/trt_bundle_experiment/convert_bundle_trt.py", line 193, in <module>
    trt_model = convert_to_trt(SAVE_PATH, BUNDLE_NAME)
  File "/home/liubin/data/trt_bundle_experiment/convert_bundle_trt.py", line 185, in convert_to_trt
    trt_model = trt_convert(model, input_shape)
  File "/home/liubin/data/trt_bundle_experiment/convert_bundle_trt.py", line 170, in trt_convert
    trt_ts_model = torch_tensorrt.compile(
  File "/opt/conda/lib/python3.8/site-packages/torch_tensorrt/_compile.py", line 125, in compile
    return torch_tensorrt.ts.compile(
  File "/opt/conda/lib/python3.8/site-packages/torch_tensorrt/ts/_compiler.py", line 136, in compile
    compiled_cpp_mod = _C.compile_graph(module._c, _parse_compile_spec(spec))
RuntimeError: Freezing modules containing prim::ModuleContainerIndex is not supported
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
 - Any other relevant information:bundle_version:0.3.5

## Additional context

<!-- Add any other context about the problem here. -->
