# YOLO ONNX Runtime Object Detector in Rust

A high-performance command-line application written in Rust for real-time object detection. It utilizes YOLO models (in ONNX format) powered by the `ort` (ONNX Runtime) engine. The application supports processing both static images and live video streams from webcams using `v4l` (Video4Linux).

## Features

* **Dual Modes**: Seamlessly switch between single image processing and live camera streaming.
* **Hardware Accelerated**: Leverages ONNX Runtime via the `ort` crate for efficient model inference.
* **Full Pipeline**: Handles automatic 640x640 resizing, channel normalization, and bounding box/text rendering.
* **COCO Dataset Support**: Pre-configured to recognize 80 standard object classes (people, vehicles, animals, electronics, etc.).
* **Robust CLI**: Built with `clap` for clean, type-safe command-line argument parsing.

## Prerequisites

Before building the project, ensure your system has the following dependencies:
1. **Rust Toolchain** (`cargo` and `rustc`)
2. **ONNX Runtime System Libraries** (required by the `ort` crate)
3. **V4L2 (Video4Linux) Development Headers** (required for webcam streaming on Linux)
4. **Font File**: A TrueType/OpenType font file named `LiberationSans-Regular.otf` must be present in the parent directory relative to your project root (e.g., `../LiberationSans-Regular.otf`).

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd your-repo-name
   ```

2. **Prepare your model:**
   Create a `data` directory and place your trained YOLO `.onnx` model inside it.
   ```bash
   mkdir -p data
   # Copy your model here (default name expected is yolo26s.onnx)
   ```

3. **Build the project:**
   ```bash
   cargo build --release
   ```

## 💻 Usage

The application provides two main subcommands: `image` and `camera`.

### 1. Process a Static Image

To detect objects in an image file and save the annotated output, run:

```bash
cargo run --release -- image --input path/to/input.jpg --output path/to/output.jpg
```

**Arguments for `image`:**
* `-i`, `--input` *(Required)*: Path to the source image file.
* `-o`, `--output` *(Required)*: Destination path where the processed image will be saved.
* `-m`, `--model` *(Optional)*: Path to the ONNX model file. Defaults to `data/yolo26s.onnx`.

---

### 2. Stream from a Webcam (Linux)

To run continuous detection on a live video stream and output real-time logs to the console, run:

```bash
cargo run --release -- camera --device /dev/video0
```

**Arguments for `camera`:**
* `-d`, `--device` *(Optional)*: Path to the V4L camera device. Defaults to `/dev/video0`.
* `-w`, `--width` *(Optional)*: Frame width configuration. Defaults to `640`.
* `--height` *(Optional)*: Frame height configuration. Defaults to `480`.
* `--model` *(Optional)*: Path to the ONNX model file. Defaults to `data/yolo26s.onnx`.

## ⚙️ Dependencies

Ensure your `Cargo.toml` includes the following external crates to support this codebase:

```toml
[dependencies]
ort = "2.0" # Adjust version based on your environment
image = "0.24"
imageproc = "0.23"
ab_glyph = "0.2"
clap = { version = "4.0", features = ["derive"] }
v4l = "0.14"
```
