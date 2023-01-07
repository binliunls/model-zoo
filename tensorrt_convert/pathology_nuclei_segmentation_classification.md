##  Description

Cannot export a model to TensorRT after successfully transfered to torchscript.

## To Reproduce

Steps to reproduce the behavior:

1. Pull the monai image 1.1.0 from [link](https://hub.docker.com/r/projectmonai/monai/tags)
1. Star a docker container with the downloaded image in step 1.
1. Run the code below.

```
import os
import ast

from monai.bundle import ConfigParser
from monai.utils.module import instantiate
import pathlib
import traceback
import glob
from typing import Union, Sequence, Tuple
from export_to_trt import load_model_and_export
import sys


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


def batch_export(root_path):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    bundle_list = [
        os.path.join(root_path, x)
        for x in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, x)) and x[0] != "_"
    ]
    pick_bundle = "pathology_nuclei_segmentation_classification"
    for bundle in bundle_list:
        bundle_name = os.path.basename(bundle)

        if pick_bundle and bundle_name != pick_bundle:
            continue
        sys.path.append(bundle)
        bundle_config_path = os.path.join(bundle, "configs")
        bundle_model_path = os.path.join(bundle, "models", "model.pt")
        trt_model_path = os.path.join(bundle, "models", "model_trt.ts")
        if os.path.exists(trt_model_path):
            continue

        print(f"==============Start to convert {bundle}.==============")
        try:
            meta_json = os.path.join(bundle_config_path, "metadata.json")
            train_files = glob.glob(os.path.join(bundle_config_path, "train.*"))
            train_json = train_files[0]
            infer_files = glob.glob(os.path.join(bundle_config_path, "inference.*"))
            infer_json = infer_files[0]
            train_json = infer_json if not os.path.exists(train_json) else train_json
            parser = ConfigParser()
            parser.read_meta(meta_json)
            parser.read_config(train_json)
            if "_meta_#network_data_format#inputs#latent" in parser:
                input_channels = parser[
                    "_meta_#network_data_format#inputs#latent#num_channels"
                ]
                input_spatial_shape = parser[
                    "_meta_#network_data_format#inputs#latent#spatial_shape"
                ]
            else:
                # meta_data = parser["_meta_"]["network_data_format"]["inputs"]["image"]
                input_channels = parser[
                    "_meta_#network_data_format#inputs#image#num_channels"
                ]
                input_spatial_shape = parser[
                    "_meta_#network_data_format#inputs#image#spatial_shape"
                ]
            spatial_shape = _get_fake_spatial_shape(input_spatial_shape)
            if not input_channels:
                spatial_shape = (1, *spatial_shape)
            else:
                spatial_shape = (1, input_channels, *spatial_shape)
            if "gnetwork" in parser:
                net_id = "gnetwork"
            elif "net" in parser:
                net_id = "net"
            else:
                net_id = "network"

            model = parser.get_parsed_content(net_id)
            # except ModuleNotFoundError:
            #     item = parser.get_parsed_content(net_id, instantiate=False)
            #     modname = item.resolve_module_name()
            #     args = item.resolve_args()
            #     model = instantiate(modname, **args)
            load_model_and_export(
                model, bundle_model_path, trt_model_path, spatial_shape
            )
        except Exception as e:
            error_info = traceback.format_exc()
            print(error_info)
            print(f"==================Failed with bundle {bundle}.==============")
        else:
            print(f"==================Done with bundle {bundle}.=================")
        print("\n\n\n")
        sys.path.remove(bundle)


if __name__ == "__main__":
    ROOT_PATH = pathlib.Path("/home/liubin/data/trt_bundle_experiment")
    batch_export(ROOT_PATH)

```
```
import argparse
import os
import torch_tensorrt
import torch

__all__ = ["load_model_and_export"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_export(
    model,
    model_name,
    out_name,
    input_size,
):
    """
    Loading a model by name.

    Args:
        model: torch.nn.module for network.
        model_name: a path name of the model that need to be loaded.
        out_name: a name for output model.
        input_size: input size for the transfered model.
    """
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    model.eval()
    with torch.no_grad():
        jit_model = torch.jit.script(model)

    inputs = [
        torch_tensorrt.Input(
            min_shape=input_size,
            opt_shape=input_size,
            max_shape=input_size,
            dtype=torch.float,
        )
    ]
    enabled_precision = {torch.float, torch.half}
    with torch_tensorrt.logging.graphs():
        trt_ts_model = torch_tensorrt.compile(
            jit_model, inputs=inputs, enabled_precisions=enabled_precision
        )
    torch.jit.save(trt_ts_model, out_name)
```

**Program stacked with output**:

```
GRAPH: [Torch-TensorRT] - Found that node  = prim::If(%15784) # /opt/monai/monai/networks/nets/hovernet.py:578:12
  block0():
     = prim::RaiseException(%24, %23) # /opt/monai/monai/networks/nets/hovernet.py:579:16
    -> ()
  block1():
    -> ()
  is an exception or pass node (EliminateChecks)
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
 - Any other relevant information:bundle_version:0.1.2

## Additional context

<!-- Add any other context about the problem here. -->
