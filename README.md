# Getting started

This library relies on the [tch](https://github.com/LaurentMazare/tch-rs) crate for bindings to the C++ Libtorch API. A more detailed installation of this library can be found at the link to this repository.

# Inference

```
use rusty_yolo;
use tch;

fn main() {
    let device = tch::Device::cuda_if_available();
    let yolo_model = rusty_yolo::YOLO::new("model.pt", 384, 640, device);
    let mut original_image = tch::vision::image::load("/images/zidane.jpg").unwrap();

    let results = yolo_model.predict(&original_image, 0.25, 0.35);
    yolo_model.draw_rectangle(&mut original_image, &results);
    tch::vision::image::save(&original_image, "images/result.jpg").unwrap();
}

```

![alt text](https://github.com/igor-yusupov/rusty-yolo/blob/main/examples/simple_inference/images/result.jpg?raw=true)
