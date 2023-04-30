use rusty_yolo;
use std::time::Instant;
use tch;

fn main() {
    let device = tch::Device::cuda_if_available();
    let yolo_model = rusty_yolo::YOLO::new(
        "./examples/simple_inference/weights/model.pt",
        384,
        640,
        device,
    );
    let mut original_image =
        tch::vision::image::load("./examples/simple_inference/images/zidane.jpg").unwrap();
    let start_time = Instant::now();

    let results = yolo_model.predict(&original_image, 0.25, 0.35);
    yolo_model.draw_rectangle(&mut original_image, &results);
    let end_time = Instant::now();
    let elapsed_time = end_time.duration_since(start_time);
    println!("Running time: {} milliseconds ⚡", elapsed_time.as_millis());
    tch::vision::image::save(
        &original_image,
        "./examples/simple_inference/images/result.jpg",
    )
    .unwrap();
}
