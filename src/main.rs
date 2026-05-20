use clap::Parser;
use opencv::{core, dnn, imgcodecs, imgproc, prelude::*, videoio};
use ort::{session::Session, value::Value};
use rayon::prelude::*;

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
    #[arg(long, default_value_t = -1)]
    camera: isize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut model = Session::builder()?.commit_from_file(args.model_path.as_str())?;

    let mut img = Mat::default();
    if args.camera >= 0 {
        let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

        if !videoio::VideoCapture::is_opened(&cam)? {
            println!("Error open cammera V4L");
            std::process::exit(-1);
        }

        cam.read(&mut img)?;
    } else {
        img = imgcodecs::imread(args.input_image.as_str(), imgcodecs::IMREAD_COLOR)?;
    }
    detect(&mut img, &args, &mut model)?;
    Ok(())
}

fn detect(
    img: &mut Mat,
    args: &Args,
    model: &mut Session,
) -> Result<(), Box<dyn std::error::Error>> {
    let img_width = img.cols() as f32;
    let img_height = img.rows() as f32;

    let mut resized = Mat::default();
    imgproc::resize(
        img,
        &mut resized,
        core::Size::new(640, 640),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let mut input_data = vec![0.0f32; 1 * 3 * 640 * 640];

    let (r_slice, rest) = input_data.split_at_mut(640 * 640);
    let (g_slice, b_slice) = rest.split_at_mut(640 * 640);

    let r_chunks = r_slice.par_chunks_mut(640);
    let g_chunks = g_slice.par_chunks_mut(640);
    let b_chunks = b_slice.par_chunks_mut(640);

    r_chunks
        .zip(g_chunks)
        .zip(b_chunks)
        .enumerate()
        .for_each(|(y, ((r_row, g_row), b_row))| {
            if let Ok(row) = resized.at_row::<core::Vec3b>(y as i32) {
                for x in 0..640 {
                    let pixel = row[x];
                    r_row[x] = pixel[2] as f32 / 255.0; // R
                    g_row[x] = pixel[1] as f32 / 255.0; // G
                    b_row[x] = pixel[0] as f32 / 255.0; // B
                }
            }
        });

    let input_tensor = Value::from_array(([1usize, 3, 640, 640], input_data))?;
    let input_value = ort::session::SessionInputValue::from(&input_tensor);

    let outputs = model.run([input_value])?;

    let output_value = &outputs[0];
    let (_shape, raw_data) = output_value.try_extract_tensor::<f32>()?;

    let target_class_id = args.class;

    let filtered_results: Vec<(core::Rect, f32)> = (0..8400)
        .into_par_iter()
        .filter_map(|i| {
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
                Some((rect, class_conf))
            } else {
                None
            }
        })
        .collect();

    let mut bboxes = core::Vector::<core::Rect>::new();
    let mut scores = core::Vector::<f32>::new();
    for (rect, conf) in filtered_results {
        bboxes.push(rect);
        scores.push(conf);
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
            img,
            rect,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;

        let label_text = format!("object {:.2}", conf);
        imgproc::put_text(
            img,
            &label_text,
            core::Point::new(rect.x, rect.y - 10),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.7,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;
        println!(
            "{}, X: {} Y: {},  Width: {} Height: {}",
            label_text, rect.x, rect.y, rect.width, rect.height
        );
    }

    let params = core::Vector::<i32>::new();
    imgcodecs::imwrite(&args.output_image.as_str(), img, &params)?;
    println!("Found {} ", indices.len());
    Ok(())
}
