use ab_glyph::{FontRef, PxScale};
use clap::{Parser, Subcommand};
use image::{Rgb, RgbImage, imageops::FilterType};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use ort::session::Session;
use std::{path::Path, time::Duration};
use v4l::{buffer::Type, io::traits::CaptureStream, prelude::*};

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

pub fn detect_on_image(
    mut original_img: RgbImage,
    session: &mut Session,
    font: &ab_glyph::FontRef,
    scale: PxScale,
) -> Result<RgbImage, Box<dyn std::error::Error>> {
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

            let text = format!("{} ({:.2})", CLASSES[class_id], score);
            draw_text_mut(
                &mut original_img,
                Rgb([255, 0, 0]), // Red color for text
                x1,
                y1 - 20, // Position text slightly above the bounding box
                scale,
                font,
                &text,
            );
            println!(
                "Detect class {} confidence {:.2} location [{}, {}, {}, {}]",
                CLASSES[class_id], score, x1, y1, x2, y2
            );
        }
    }

    Ok(original_img)
}

pub fn run_detection(
    image_path: &str,
    model_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // initialize ONNX Runtime
    let mut session = Session::builder()?.commit_from_file(model_path)?;
    let original_img = image::open(Path::new(image_path))?.to_rgb8();

    // Load a font
    let font_data = include_bytes!("../LiberationSans-Regular.otf");
    let font = FontRef::try_from_slice(font_data as &[u8]).expect("Error loading font");
    let scale = PxScale::from(20.0);

    let processed = detect_on_image(original_img, &mut session, &font, scale)?;
    processed.save(output_path)?;
    println!("Image save to: {}", output_path);

    Ok(())
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Image {
        #[arg(short, long)]
        input: String,
        #[arg(short, long)]
        output: String,
        #[arg(short, long, default_value = "data/yolo26s.onnx")]
        model: String,
    },
    Camera {
        #[arg(short, long, default_value = "/dev/video0")]
        device: String,
        #[arg(short, long, default_value = "640")]
        width: u32,
        #[arg(long, default_value = "480")]
        height: u32,
        #[arg(long, default_value = "data/yolo26s.onnx")]
        model: String,
    },
}

fn run_camera(
    device_path: &str,
    _width: u32,
    _height: u32,
    model_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::with_path(device_path)?;
    let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 3)?;
    let mut session = Session::builder()?.commit_from_file(model_path)?;
    let font_data = include_bytes!("../LiberationSans-Regular.otf");
    let font = FontRef::try_from_slice(font_data as &[u8]).expect("Error loading font");
    let scale = PxScale::from(20.0);

    println!("Starting camera loop...");
    loop {
        let (data, _) = stream.next()?;
        // let img = RgbImage::from_raw(width, height, data.to_vec())
        //     .ok_or("Failed to create image from buffer")?;
        let img = image::load_from_memory_with_format(data, image::ImageFormat::Jpeg)?.to_rgb8();

        let _processed = detect_on_image(img, &mut session, &font, scale)?;
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Image {
            input,
            output,
            model,
        } => {
            if let Err(e) = run_detection(&input, &model, &output) {
                eprintln!("Error: {}", e);
            }
        }
        Commands::Camera {
            device,
            width,
            height,
            model,
        } => {
            if let Err(e) = run_camera(&device, width, height, &model) {
                eprintln!("Error: {}", e);
            }
        }
    }
}
