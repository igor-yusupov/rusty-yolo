#adapted from export.py in YOLOv5 🚀 by Ultralytics, AGPL-3.0 license

import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch import nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.yolo import Detect
from utils.general import (LOGGER, Profile, check_img_size,
                           colorstr, file_size, get_default_args, url2file)
from utils.torch_utils import select_device, smart_inference_mode



def try_export(inner_func):
    # YOLOv5 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f'{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            return None, None

    return outer_func


@try_export
def export_torchscript(model, im, d, prefix=colorstr('TorchScript:')):
    # YOLOv5 TorchScript model export
    LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
    f:str = 'yolo.torchscript'
    ts = torch.jit.trace(model, im, strict=False)
    extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
    ts.save(f, _extra_files=extra_files)
    return f, None


@smart_inference_mode()
def run(
        
        weights= ROOT / 'yolov5s.pt',  # weights path
        imgsz=(384, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('torchscript'),  # include formats
        inplace=False,  # set YOLOv5 Detect() inplace=True
        dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
    
):
    include = [x.lower() for x in include]  # to lowercase
    file = weights  # PyTorch weights

    class TorchScriptWithoutTuple(nn.Module):
        def __init__(self) -> None:
            super().__init__()
        

        def forward(self, x):
            return x[0]
    
    # Load PyTorch model
    device = select_device(device)
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model
    
    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand


    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for _, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs

    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")
    d = {'shape': im.shape, 'stride': int(max(model.stride)), 'names': model.names}
    # Exports
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    model =  nn.Sequential(model,TorchScriptWithoutTuple())
    export_torchscript(model, im, d)
   



if __name__ == '__main__':
  run()
