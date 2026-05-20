# YOLO ONNX Runtime Object Detector in Rust

A lightweight Rust application that runs YOLO object detection using ONNX Runtime (`ort`) and OpenCV for image processing.

## Features

* **ONNX Runtime:** High-performance inference via `ort` crate.
* **OpenCV Integration:** Handles image resizing, color conversion, and drawing.
* **CLI Interface:** Easy configuration using `clap`.
* **NMS Filtering:** Built-in Non-Maximum Suppression via OpenCV DNN module.

## Dependencies

Ensure your system has the required native libraries installed:
* OpenCV 4.x
* ONNX Runtime

Add these to your `Cargo.toml`:

```toml
[dependencies]
clap = { version = "4.0", features = ["derive"] }
opencv = "0.92"
ort = "2.0"
```

## Usage

```bash
cargo run -- \
  --model-path path/to/yolov8n.onnx \
  --input-image input.jpg \
  --output-image output.jpg \
  --class 0 \
  --conf-threshold 0.3 \
  --nms 0.45
```

### CLI Arguments

* `-m, --model-path <PATH>`: Path to the `.onnx` model file.
* `-i, --input-image <PATH>`: Path to the input image.
* `-o, --output-image <PATH>`: Path to save the processed image.
* `--class <INT>`: Target COCO class ID to detect (e.g., `0` for person). Default: `0`.
* `-c, --conf-threshold <FLOAT>`: Confidence threshold. Default: `0.3`.
* `-n, --nms <FLOAT>`: Non-maximum suppression threshold. Default: `0.45`.
* `--camera <INT>`: Use Linux camera /dev/video*. Default: `-1`.
## How It Works

1. **Preprocessing:** Resizes input image to 640x640 and normalizes pixels to `[0.0, 1.0]` CHW format.
2. **Inference:** Passes the tensor to ONNX Runtime.
3. **Postprocessing:** Extracts bounding boxes for the selected class ID.
4. **NMS:** Filters overlapping boxes.
5. **Visualization:** Draws green bounding boxes and labels on the original image.

```
const CLASSES: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

```
