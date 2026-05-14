use clap::Parser;
use opencv::{core, dnn, imgcodecs, imgproc, prelude::*};
use ort::{session::Session, value::Value};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    model_path: String,
    #[arg(short, long)]
    input_image: String,
    #[arg(short, long)]
    output_image: String,
    #[arg(long, default_value_t = 0, help = "0..80 see documentation")]
    class: usize,
    #[arg(short, long, default_value_t = 0.3)]
    conf_threshold: f32,
    #[arg(long, default_value_t = 0.45)]
    nms: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut model = Session::builder()?.commit_from_file(args.model_path.as_str())?;

    let mut img = imgcodecs::imread(args.input_image.as_str(), imgcodecs::IMREAD_COLOR)?;
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

    let target_class_id = args.class;

    let mut bboxes = core::Vector::<core::Rect>::new();
    let mut scores = core::Vector::<f32>::new();

    for i in 0..8400 {
        let class_idx = (4 + target_class_id) * 8400 + i;
        let class_conf = raw_data[class_idx];

        if class_conf >= args.conf_threshold {
            let cx = raw_data[0 * 8400 + i];
            let cy = raw_data[1 * 8400 + i];
            let w = raw_data[2 * 8400 + i];
            let h = raw_data[3 * 8400 + i];

            let x1 = (((cx - w / 2.0) / 640.0) * img_width) as i32;
            let y1 = (((cy - h / 2.0) / 640.0) * img_height) as i32;
            let x2 = (((cx + w / 2.0) / 640.0) * img_width) as i32;
            let y2 = (((cy + h / 2.0) / 640.0) * img_height) as i32;

            let rect = core::Rect::new(x1, y1, x2 - x1, y2 - y1);
            bboxes.push(rect);
            scores.push(class_conf);
        }
    }

    let mut indices = core::Vector::<i32>::new();
    dnn::nms_boxes(
        &bboxes,
        &scores,
        args.conf_threshold,
        args.nms,
        &mut indices,
        1.0,
        0,
    )?;

    for idx in indices.iter() {
        let rect = bboxes.get(idx as usize)?;
        let conf = scores.get(idx as usize)?;

        imgproc::rectangle(
            &mut img,
            rect,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;

        let label_text = format!("person {:.2}", conf);
        imgproc::put_text(
            &mut img,
            &label_text,
            core::Point::new(rect.x, rect.y - 10),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.7,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;
    }

    let params = core::Vector::<i32>::new();
    imgcodecs::imwrite(&args.output_image.as_str(), &img, &params)?;
    println!("Found {} ", indices.len());
    Ok(())
}
