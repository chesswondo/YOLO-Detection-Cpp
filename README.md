# YOLOv8 Inference Benchmark: Python vs C++

This repository contains a performance comparison of YOLOv8 inference implementations using **Python (Ultralytics)** and **C++ (ONNX Runtime)**.

My goal was to analyze the overhead introduced by the Python interpreter, Global Interpreter Lock (GIL), Garbage Collector and Pytorch eager mode compared to a raw C++ .onnx implementation in a real-time computer vision context.


## Project Structure
<pre>
├── YoloCpp/                      # C++ implementation
│   ├── src/
│   |    ├── main.cpp             # Entry point, async loop, metrics
│   |    ├── YoloDetector.h
│   |    └── YoloDetector.cpp
│   └── CMakeLists.txt            # Build configuration
│
├── YoloPython/                   # Python implementation
│   ├── src/
│   |    ├── fps_counter.py       # Separate fps counter on a single image (inference only)
│   |    ├── model_export.py      # Script for exporting the model to .onnx
│   |    └── web_inference.py     # Benchmark script with threading
│   └── requirements.txt          # Python dependencies
│
└── README.md
</pre>


## Benchmark Results

In this project I measure not just FPS, but also:
- **Inference Latency:** Pure model computation time.
- **System Latency (End-to-End):** Time from frame capture to visualization.
- **Jitter (standard deviation):** Stability of the processing pipeline.

Tests were conducted on a webcam feed (640x480). The models were configured with identical confidence thresholds (0.5).

| Metric | Python (Ultralytics) | C++ (ONNX Runtime) |
| :--- | :--- | :--- |
| **Inference Time (Avg)** | ~13.1 ms | ~10.6 ms |
| **Inference Jitter (Std)** | ~2.8 ms | ~1.1 ms |
| **End-to-End Latency (Avg)** | ~21.0 ms | ~15.6 ms |
| **End-to-End Jitter (Std)** | ~7.5 ms | ~2.0 ms |
| **Real FPS** | ~48 FPS | ~64 FPS |
| **RAM utilization** | ~980 MB | ~700 MB |

**Observation:** While C++ slightly (~25% faster) outperforms Python in pure GPU inference time (+ pre/postprocessing CPU operations) and overall throughput (real FPS), the difference in system stability (Jitter) is significant due to the lack of GC overhead and faster pre/post-processing. It is critical for edge devices.


## Methodology
**1. Warmup:** A dummy tensor is passed through the network to initialize CUDA contexts and allocate memory buffers.

**2. Threaded Capture:** A separate thread constantly grabs the latest frame from the camera buffer.

**3. Measurement:**
- Inference Time: Run() method execution time.
- Frame Time: Time between loop iterations (includes visualization imshow overhead).
