First, you have to create an export of weights in torchscript format. 

1. Clone the Yolov5 repository into the directory where you would like to store it.
>> git clone https://github.com/ultralytics/yolov5.git

2. Move [export_torchscript.py](export_torchscript.py) into the Yolov5 directory.

3. If you intend to use a different image size (to the one used in the example) you will need to specify this* in line 64 of [export_torchscript.py](export_torchscript.py) 

4. Run the export script. This should export a file called "yolo.torchscript" into the yolov5 directory.

3. Move the exported torchscript file to [/models](../models/)

4. Update the *"weights"* variable in [/examples/simple_inference/main.rs](/examples/simple_inference/main.rs) to refer to the weights file you want to use.

5. If you changed the image size, you'll also need to update the 


*By default Yolov5 works best with an image size of 640x640 and the authors recommend padding with zeros to bring smaller dimensions up to size (e.g. padding 640x480 up to 640x640).