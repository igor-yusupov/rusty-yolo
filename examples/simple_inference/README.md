
This example demonstrates how to run the yolov5 inference.

First, you have to create an export of weights in torchscript format. This can be done using the https://github.com/ultralytics/yolov5 repository by running the command:
`python export.py --imgsz 384 640`

Then you need to get rid of the tuple format on the model output, you can do this with a script:
```
import torch
import torch.nn as nn


class TorchScriptWithoutTuple(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]

def main():
    model = torch.hub.load('ultralytics/yolov5', 'custom', "yolov5s.torchscript").eval()
    save_model = TorchScriptWithoutTuple(model)
    traced_model = torch.jit.trace(save_model, (torch.rand((1, 3, 384, 640))))
    torch.jit.save(traced_model, "weights/model.pt",)


if __name__ == "__main__":
    main()

```
