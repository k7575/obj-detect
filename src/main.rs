use opencv::{core, imgcodecs, imgproc, prelude::*};
use ort::{session::Session, value::Value};

const MODEL_PATH: &str = "data/yolov8s.onnx";
const INPUT_IMAGE: &str = "data/image.jpg";
const OUTPUT_IMAGE: &str = "data/output.jpg";
const CONF_THRESHOLD: f32 = 0.3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Session::builder()?.commit_from_file(MODEL_PATH)?;

    let mut img = imgcodecs::imread(INPUT_IMAGE, imgcodecs::IMREAD_COLOR)?;
    let img_width = img.cols() as f32;
    let img_height = img.rows() as f32;

    let mut resized = Mat::default();
    imgproc::resize(
        &img,
        &mut resized,
        core::Size::new(640, 640),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let mut input_data = vec![0.0f32; 1 * 3 * 640 * 640];

    for y in 0..640 {
        for x in 0..640 {
            let pixel = resized.at_2d::<core::Vec3b>(y, x)?;
            let r_idx = 0 * 640 * 640 + y as usize * 640 + x as usize;
            let g_idx = 1 * 640 * 640 + y as usize * 640 + x as usize;
            let b_idx = 2 * 640 * 640 + y as usize * 640 + x as usize;

            input_data[r_idx] = pixel[2] as f32 / 255.0; // R
            input_data[g_idx] = pixel[1] as f32 / 255.0; // G
            input_data[b_idx] = pixel[0] as f32 / 255.0; // B
        }
    }

    let input_tensor = Value::from_array(([1usize, 3, 640, 640], input_data))?;
    let input_value = ort::session::SessionInputValue::from(&input_tensor);

    let outputs = model.run([input_value])?;

    let output_value = &outputs[0];
    let (_shape, raw_data) = output_value.try_extract_tensor::<f32>()?;

    let mut person_count = 0;
    let target_class_id = 0; // 0 — 'person'

    for i in 0..8400 {
        let class_idx = (4 + target_class_id) * 8400 + i;
        let class_conf = raw_data[class_idx];

        if class_conf >= CONF_THRESHOLD {
            person_count += 1;

            let cx = raw_data[0 * 8400 + i];
            let cy = raw_data[1 * 8400 + i];
            let w = raw_data[2 * 8400 + i];
            let h = raw_data[3 * 8400 + i];

            let x1 = (((cx - w / 2.0) / 640.0) * img_width) as i32;
            let y1 = (((cy - h / 2.0) / 640.0) * img_height) as i32;
            let x2 = (((cx + w / 2.0) / 640.0) * img_width) as i32;
            let y2 = (((cy + h / 2.0) / 640.0) * img_height) as i32;

            imgproc::rectangle(
                &mut img,
                core::Rect::new(x1, y1, x2 - x1, y2 - y1),
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;

            let label_text = format!("person {:.2}", class_conf);
            imgproc::put_text(
                &mut img,
                &label_text,
                core::Point::new(x1, y1 - 10),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.7,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                false,
            )?;
        }
    }

    let params = core::Vector::<i32>::new();
    imgcodecs::imwrite(OUTPUT_IMAGE, &img, &params)?;
    println!("Found {} persons.", person_count);

    Ok(())
}
