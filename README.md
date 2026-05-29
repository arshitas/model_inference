# 🚀 End-to-End Model Optimization & Deployment

**PyTorch → ONNX → TensorRT → Triton Inference Server**

A complete inference optimization pipeline that trains a ResNet-18 image classifier on CIFAR-10, exports it through multiple optimization stages, and deploys it on NVIDIA Triton Inference Server — benchmarking latency improvements at every step.

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Benchmark Results](#benchmark-results)
- [Triton Deployment](#triton-deployment)
- [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────┐     ┌────────────────────┐     ┌─────────────────────┐
│  CIFAR-10   │────▶│ PyTorch  │────▶│      ONNX          │────▶│     TensorRT        │
│  Dataset    │     │ ResNet-18│     │  (Portable IR)     │     │  (FP32 / FP16)      │
└─────────────┘     └────┬─────┘     └────────┬───────────┘     └──────────┬──────────┘
                         │                    │                            │
                         │ TorchScript        │ ONNX Runtime               │ TRT Engine
                         ▼                    ▼                            ▼
                   ┌─────────────────────────────────────────────────────────────┐
                   │              NVIDIA Triton Inference Server                 │
                   │  ┌────────────┐  ┌────────────┐  ┌───────────────────┐     │
                   │  │  PyTorch   │  │    ONNX    │  │    TensorRT       │     │
                   │  │  Backend   │  │   Backend  │  │    Backend        │     │
                   │  └────────────┘  └────────────┘  └───────────────────┘     │
                   │            gRPC :8001  /  HTTP :8000                        │
                   └─────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │  Client App  │
                                   │  (benchmark) │
                                   └──────────────┘
```

---

## 📁 Project Structure

```
Inference_Project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/
│   ├── train.py                       # Train ResNet-18 on CIFAR-10
│   ├── export_onnx.py                 # Export to ONNX + validation
│   ├── optimize_tensorrt.py           # Build TensorRT FP32/FP16 engines
│   ├── benchmark.py                   # Latency & throughput benchmarks
│   └── triton_client.py               # Triton gRPC client
├── scripts/
│   ├── run_pipeline.py                # Full pipeline orchestrator
│   └── start_triton.sh                # Launch Triton via Docker
├── models/                            # Generated model artifacts
│   ├── cifar10_resnet18.pth           # PyTorch state dict
│   ├── cifar10_resnet18_ts.pt         # TorchScript traced model
│   ├── cifar10_resnet18.onnx          # ONNX model
│   ├── cifar10_resnet18_fp32.engine   # TensorRT FP32 engine
│   ├── cifar10_resnet18_fp16.engine   # TensorRT FP16 engine
│   └── benchmark_results.json         # Benchmark data
└── triton_model_repo/                 # Triton model repository
    ├── cifar10_pytorch/
    │   ├── config.pbtxt
    │   └── 1/model.pt
    ├── cifar10_onnx/
    │   ├── config.pbtxt
    │   └── 1/model.onnx
    └── cifar10_tensorrt/
        ├── config.pbtxt
        └── 1/model.plan
```

---

## ⚙️ Prerequisites

| Requirement            | Version    | Notes                              |
|------------------------|------------|------------------------------------|
| Python                 | ≥ 3.9     | 3.10+ recommended                  |
| PyTorch                | ≥ 2.0     | With CUDA support                  |
| ONNX Runtime GPU       | ≥ 1.15    | `pip install onnxruntime-gpu`      |
| TensorRT               | ≥ 8.6     | System install or `pip`            |
| NVIDIA GPU             | CC ≥ 7.0  | Volta / Turing / Ampere / Hopper   |
| Docker + nvidia-docker | Latest     | For Triton server                  |
| CUDA Toolkit           | ≥ 11.8    | Matching TensorRT version          |

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Option 1: Run the full pipeline (recommended)

```bash
python scripts/run_pipeline.py
```

This will:
1. Download CIFAR-10 and train ResNet-18 (~5 epochs, ~93% accuracy)
2. Export to ONNX with numerical validation
3. Build TensorRT FP32 & FP16 engines
4. Run benchmarks across all backends
5. Copy artifacts to the Triton model repository

### Option 2: Run each step individually

```bash
# Step 1: Train
python src/train.py --epochs 5

# Step 2: Export to ONNX
python src/export_onnx.py

# Step 3: Build TensorRT engines
python src/optimize_tensorrt.py

# Step 4: Benchmark
python src/benchmark.py --batch-size 1

# Step 5: Deploy to Triton
bash scripts/start_triton.sh          # In terminal 1
python src/triton_client.py --benchmark  # In terminal 2
```

---

## 🔬 Pipeline Stages

### Stage 1: Training (`src/train.py`)

- **Model**: ResNet-18 (pretrained on ImageNet, fine-tuned for CIFAR-10)
- **Dataset**: CIFAR-10 (60K images, 10 classes, resized to 224×224)
- **Training**: Adam optimizer, cosine annealing LR, 5 epochs
- **Expected accuracy**: >93% validation accuracy
- **Outputs**: `models/cifar10_resnet18.pth` (state dict) + `models/cifar10_resnet18_ts.pt` (TorchScript)

### Stage 2: ONNX Export (`src/export_onnx.py`)

- **Format**: ONNX opset 17 with dynamic batch axis
- **Validation**:
  - `onnx.checker.check_model()` graph validation
  - Numerical diff < 1e-4 vs PyTorch (random input sanity check)
- **Output**: `models/cifar10_resnet18.onnx`

### Stage 3: TensorRT Optimization (`src/optimize_tensorrt.py`)

- **Engines built**:
  - FP32 — baseline TensorRT precision
  - FP16 — mixed-precision for maximum throughput
- **Dynamic shapes**: batch 1 → 64 (optimized for batch 8)
- **Validation**: TensorRT output vs ONNX Runtime (max diff check)
- **Outputs**: `models/cifar10_resnet18_fp32.engine`, `models/cifar10_resnet18_fp16.engine`

### Stage 4: Benchmarking (`src/benchmark.py`)

Measures latency and throughput for all five backends using GPU-synchronized timing:

| Metric        | Description                             |
|---------------|-----------------------------------------|
| Mean latency  | Average inference time per batch (ms)   |
| p50 / p95 / p99 | Latency percentiles                  |
| Throughput    | Images processed per second             |
| Speedup       | Relative to PyTorch eager baseline      |

---

## 📊 Benchmark Results

> **Expected results** on an NVIDIA A100 GPU (batch size = 1):

| Backend          | Mean (ms) | p95 (ms) | Throughput (img/s) | Speedup |
|------------------|-----------|-----------|--------------------|---------|
| PyTorch Eager    | ~5.2      | ~5.8      | ~192               | 1.00×   |
| TorchScript      | ~4.8      | ~5.3      | ~208               | 1.08×   |
| ONNX Runtime     | ~2.1      | ~2.4      | ~476               | 2.48×   |
| TensorRT FP32    | ~1.3      | ~1.5      | ~769               | 4.00×   |
| TensorRT FP16    | ~0.7      | ~0.9      | ~1429              | 7.43×   |

> *Actual numbers vary by GPU. Run `python src/benchmark.py` for your hardware.*

### Key Observations

- **ONNX Runtime** provides ~2.5× speedup from graph optimizations (constant folding, operator fusion)
- **TensorRT FP32** adds another ~2× from kernel auto-tuning and layer fusion
- **TensorRT FP16** nearly doubles FP32 throughput via mixed-precision inference with <0.1% accuracy drop
- The full pipeline delivers **~7× latency reduction** from PyTorch eager to TensorRT FP16

---

## 🖥️ Triton Deployment

### Start Triton Server

```bash
# Run the pipeline first to populate model artifacts
python scripts/run_pipeline.py

# Start Triton (requires Docker + NVIDIA Container Toolkit)
bash scripts/start_triton.sh
```

### Query Models

```bash
# Health check + single inference on all models
python src/triton_client.py --url localhost:8001

# Benchmark a specific model
python src/triton_client.py --model cifar10_tensorrt --benchmark --count 500

# Different batch size
python src/triton_client.py --model cifar10_onnx --benchmark --batch-size 8
```

### Triton Features Configured

- **Dynamic batching**: Groups incoming requests into batches of 4/8/16 for higher throughput
- **Multi-backend**: Same model served via PyTorch, ONNX, and TensorRT simultaneously
- **GPU acceleration**: All backends pinned to GPU 0
- **Metrics endpoint**: `http://localhost:8002/metrics` (Prometheus format)

---

## 🔧 Troubleshooting

| Issue                          | Solution                                                |
|--------------------------------|---------------------------------------------------------|
| `CUDA out of memory`          | Reduce `--batch-size` or free GPU memory                |
| `TensorRT build fails`        | Ensure TensorRT version matches CUDA toolkit            |
| `ONNX opset not supported`    | Try `--opset 13` for older ONNX Runtime versions        |
| `Triton model not loading`    | Check `config.pbtxt` input/output names match the model |
| `pycuda not found`            | `pip install pycuda` (requires CUDA toolkit headers)    |

---

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 📄 License

This project is for educational and demonstration purposes.
