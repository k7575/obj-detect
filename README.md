# YOLOv8 Person Detector in Rust

A lightweight command-line utility for detecting people in images using a YOLOv8 ONNX model. The application processes inputs through the ONNX Runtime (`ort`) and filters predictions using OpenCV's Non-Maximum Suppression (NMS).

## 🚀 Features

*   **YOLOv8 Inference**: Runs object detection directly from standard `.onnx` model files.
*   **Duplicate Filtering**: Uses OpenCV's NMS algorithm to prevent overlapping bounding boxes on the same individual.
*   **CLI Interface**: Configurable thresholds, model paths, and image paths via command-line arguments.

## 📋 Prerequisites

Ensure your system has the required native C++ dependencies installed before compiling:

*   **OpenCV**: Version 4.x or higher.
*   **ONNX Runtime**: Needed to link the `ort` crate.

## 🛠️ Installation & Setup

1. Add the following dependencies to your `Cargo.toml` file:

```toml
[dependencies]
clap = { version = "4.0", features = ["derive"] }
opencv = "0.93"
ort = "2.0"
```

2. Download a standard YOLOv8 export model (e.g., `yolov8s.onnx`) and save it to your local environment.

## 💻 Usage

Run the program via `cargo run` while passing the required flags.

### Basic command:
```bash
cargo run -- -m path/to/yolov8s.onnx -i path/to/input.jpg -o path/to/output.jpg
```

### Advanced configurations with thresholds:
```bash
cargo run -- \
  --model-path data/yolov8s.onnx \
  --input-image data/scene.jpg \
  --output-image data/result.jpg \
  --conf-threshold 0.4 \
  --nms 0.5
```

### Argument Reference


| Long Flag | Short Flag | Default Value | Description |
| :--- | :---: | :---: | :--- |
| `--model-path` | `-m` | *Required* | Path to the exported `yolov8s.onnx` file |
| `--input-image` | `-i` | *Required* | Path to the source image file (`.jpg`, `.png`) |
| `--output-image` | `-o` | *Required* | Path where the annotated output image will be saved |
| `--conf-threshold`| `-c` | `0.3` | Score threshold to filter out low-confidence objects |
| `--nms` | - | `0.45` | Overlap threshold used by Non-Maximum Suppression |

## ⚙️ How It Works

1. **CLI Parsing**: `clap::Parser` reads and validates arguments.
2. **Preprocessing**: The target image is resized to `640x640` and transposed into a normalized NCHW `f32` tensor.
3. **Inference**: The `ort` session feeds the input data into the model to extract raw bounding box predictions.
4. **NMS Filtering**: Matches predicting the `person` class (index `0`) are sent to `opencv::dnn::nms_boxes` to prune duplicate overlaps.
5. **Rendering**: Bounding boxes and confidence text scores are drawn on top of the original image layout.

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
