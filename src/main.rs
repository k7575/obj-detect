use image::{Rgb, imageops::FilterType};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ort::session::Session;
use std::path::Path;

const CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

pub fn run_detection(image_path: &str, model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 1. initialize ONNX Runtime
    let mut session = Session::builder()?.commit_from_file(model_path)?;

    let mut original_img = image::open(Path::new(image_path))?.to_rgb8();
    let (img_w, img_h) = original_img.dimensions();

    // YOLO 640x640
    let resized_img = image::imageops::resize(&original_img, 640, 640, FilterType::Triangle);

    let mut input_floats = vec![0.0f32; 1 * 3 * 640 * 640];

    for y in 0..640 {
        for x in 0..640 {
            let pixel = resized_img.get_pixel(x as u32, y as u32);

            let r_index = 0 * 640 * 640 + y * 640 + x;
            let g_index = 1 * 640 * 640 + y * 640 + x;
            let b_index = 2 * 640 * 640 + y * 640 + x;

            input_floats[r_index] = pixel[0] as f32 / 255.0;
            input_floats[g_index] = pixel[1] as f32 / 255.0;
            input_floats[b_index] = pixel[2] as f32 / 255.0;
        }
    }

    let input_value = ort::value::Value::from_array((vec![1, 3, 640, 640], input_floats))?;
    let input_tensor = ort::inputs!["images" => input_value];
    let outputs = session.run(input_tensor)?;

    let output_name = "output0";
    let output_view = outputs[output_name].try_extract_tensor::<f32>()?;
    let output_data = output_view.1;

    let scale_x = img_w as f32 / 640.0;
    let scale_y = img_h as f32 / 640.0;

    let green_color = Rgb([0, 255, 0]);

    for chunk in output_data.chunks_exact(6) {
        let score = chunk[4];

        if score > 0.45 {
            let x1 = (chunk[0] * scale_x) as i32;
            let y1 = (chunk[1] * scale_y) as i32;
            let x2 = (chunk[2] * scale_x) as i32;
            let y2 = (chunk[3] * scale_y) as i32;
            let class_id = chunk[5] as usize;

            let width = (x2 - x1).max(1);
            let height = (y2 - y1).max(1);

            let rect = Rect::at(x1, y1).of_size(width as u32, height as u32);

            draw_hollow_rect_mut(&mut original_img, rect, green_color);

            println!(
                "Detect class {} confidence {:.2} location [{}, {}, {}, {}]",
                CLASSES[class_id], score, x1, y1, x2, y2
            );
        }
    }

    let output_path = "output.jpg";
    original_img.save(output_path)?;
    println!("Image save to: {}", output_path);

    Ok(())
}

fn main() {
    if let Err(e) = run_detection("input.jpg", "data/yolo26s.onnx") {
        eprintln!("Error: {}", e);
    }
}
