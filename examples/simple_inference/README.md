
This example demonstrates how to run the Yolov5 inference.

1. Clone this repository to the desired location on you computer.
    >> git clone

2. Build and run the example using:

    >> cargo run --example simple_inference

3. This will perform inference on the image defined in line 15 of [main.rs](main.rs), using the Yolo torchscrpt file defined  in line 7.

4. The result will be output to the [images](images/) folder once complete, as shown on line 25 of [main.rs](main.rs).


For convenience, a copy a the Yolov5s model in torchscript format has been proivided in the [models](../../models/) folder. By deault this is where the example in [main.rs](main.rs) will look for the model. This can be changed by defining the path to a different Yolo torchscript file in the *"weights"* variable in line 7 of [main.rs](main.rs).

Several different versions of Yolov5 are available, with varying degrees of speed and accuracy. The TCH crate expects a slightly different format to what is provided by Yolo's built in export funtion. As such, please follow the **instructions** provided [here](../../models/) to obtain torchscrpt files for other versions of yolo.
