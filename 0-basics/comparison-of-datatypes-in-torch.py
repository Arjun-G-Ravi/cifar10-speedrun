"""
Benchmark speed and memory across popular PyTorch dtypes on an Ampere GPU (e.g. RTX 3090).

Covered:
Training (forward+backward):
  - FP32 (with TF32 math on Ampere)
  - AMP FP16 (autocast + GradScaler)
  - AMP BF16 (if supported)
Inference (forward only, weights actually converted / quantized):
  - FP32
  - FP16
  - BF16 (if supported)
  - INT8 weight-only (optional: bitsandbytes if installed)
  - 4-bit (NF4) weight-only (optional: bitsandbytes if installed)

Notes:
- Native int8 training on GPU is not standard; we treat int8/4-bit as inference-only weight-only quantization.
- bitsandbytes provides fast memory savings; real speedups depend on kernel support and batch size.
- Dynamic/static quantization from torch.ao.quantization mainly targets CPU; omitted here for GPU focus.
- Parameter memory for AMP training remains FP32 (master weights), so reductions only show up in inference section.

Run:
  python benchmark_dtypes_extended.py

Optional installs for extra rows:
  pip install bitsandbytes
"""

import time
import math
import torch
import torch.nn as nn
import bitsandbytes as bnb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model / workload sizes (adjust to stress GPU; keep divisible by 8 for tensor cores)
BATCH = 512
IN_FEATURE = 4096
HIDDEN1 = 8192
HIDDEN2 = 4096
OUT_FEATURE = 1000

WARMUP_STEPS_TRAIN = 8
MEASURE_STEPS_TRAIN = 40
WARMUP_STEPS_INF = 10
MEASURE_STEPS_INF = 100
TORCH_SEED = 2024

torch.manual_seed(TORCH_SEED)
if DEVICE == "cuda":
    # Ensure TF32 on for FP32 matmuls (Ampere default: "high" or "medium")
    torch.set_float32_matmul_precision("high")  # medium/high/highest
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def make_model():
    return nn.Sequential(
        nn.Linear(IN_FEATURE, HIDDEN1, bias=True),
        nn.ReLU(),
        nn.Linear(HIDDEN1, HIDDEN2, bias=True),
        nn.ReLU(),
        nn.Linear(HIDDEN2, OUT_FEATURE, bias=True)
    )

def param_size_bytes(model: nn.Module):
    return sum(p.numel() * p.element_size() for p in model.parameters())

def format_bytes(n):
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024
        i += 1
    return f"{f:.2f} {units[i]}"

def benchmark_training(mode_name, autocast_dtype=None, use_grad_scaler=False):
    model = make_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    x = torch.randn(BATCH, IN_FEATURE, device=DEVICE)
    target = torch.randn(BATCH, OUT_FEATURE, device=DEVICE)

    scaler = torch.amp.GradScaler(enabled=use_grad_scaler)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warmup
    for _ in range(WARMUP_STEPS_TRAIN):
        optimizer.zero_grad(set_to_none=True)
        if autocast_dtype:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                out = model(x)
                loss = loss_fn(out, target)
        else:
            out = model(x)
            loss = loss_fn(out, target)
        if use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(MEASURE_STEPS_TRAIN):
        optimizer.zero_grad(set_to_none=True)
        if autocast_dtype:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                out = model(x)
                loss = loss_fn(out, target)
        else:
            out = model(x)
            loss = loss_fn(out, target)
        if use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        peak_mem = 0
    elapsed = time.time() - t0
    return {
        "mode": mode_name,
        "time_total_s": elapsed,
        "steps": MEASURE_STEPS_TRAIN,
        "time_per_step_ms": (elapsed / MEASURE_STEPS_TRAIN) * 1000,
        "peak_memory": peak_mem,
        "param_master_bytes": param_size_bytes(model),
    }

def forward_passes(model, x, steps, autocast_dtype=None):
    # Warmup done outside optionally
    for _ in range(steps):
        if autocast_dtype:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                _ = model(x)
        else:
            _ = model(x)

def benchmark_inference(mode_name, build_fn, input_dtype=None, autocast_dtype=None):
    # build_fn returns a prepared (converted / quantized) model on DEVICE
    model = build_fn()
    model.eval()
    torch.set_grad_enabled(False)

    if input_dtype is None:
        # Use model's first layer weight dtype (raw dtype or underlying .weight)
        any_param = next(model.parameters())
        input_dtype = any_param.dtype

    x = torch.randn(BATCH, IN_FEATURE, device=DEVICE, dtype=input_dtype)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warmup
    forward_passes(model, x, WARMUP_STEPS_INF, autocast_dtype=autocast_dtype)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    forward_passes(model, x, MEASURE_STEPS_INF, autocast_dtype=autocast_dtype)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        peak_mem = 0

    elapsed = time.time() - t0
    param_bytes = param_size_bytes(model)

    return {
        "mode": mode_name,
        "time_total_s": elapsed,
        "steps": MEASURE_STEPS_INF,
        "time_per_step_ms": (elapsed / MEASURE_STEPS_INF) * 1000,
        "peak_memory": peak_mem,
        "param_bytes": param_bytes,
    }

def try_bitsandbytes_linear_model(nbits="8"):
    """
    Build weight-only quantized model using bitsandbytes Linear layers.
    nbits: "8" or "4"
    For 4-bit we use nf4 (if available) for better accuracy.
    """
    try:
        import bitsandbytes as bnb
    except Exception:
        return None, f"bitsandbytes not installed (pip install bitsandbytes) for {nbits}-bit."

    # Choose appropriate layer class
    if nbits == "8":
        LinearCls = bnb.nn.Linear8bitLt
        kwargs = dict(has_fp16_weights=False)
    elif nbits == "4":
        # 4-bit quantization; nf4 or fp4 (nf4 preferred)
        LinearCls = bnb.nn.Linear4bit
        kwargs = dict(compute_dtype=torch.float16, quant_type="nf4")
    else:
        return None, "Unsupported bit width."

    class QNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                LinearCls(IN_FEATURE, HIDDEN1, bias=True, **kwargs),
                nn.ReLU(),
                LinearCls(HIDDEN1, HIDDEN2, bias=True, **kwargs),
                nn.ReLU(),
                LinearCls(HIDDEN2, OUT_FEATURE, bias=True, **kwargs),
            )
        def forward(self, x):
            return self.seq(x)

    model = QNet().to(DEVICE)
    # Initialize (the layers already allocate quantized states)
    return model, None

def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print()

    training_results = []
    if DEVICE == "cuda":
        training_results.append(benchmark_training("train_fp32(TF32)"))
        training_results.append(benchmark_training("train_amp_fp16", autocast_dtype=torch.float16, use_grad_scaler=True))
        if torch.cuda.is_bf16_supported():
            training_results.append(benchmark_training("train_amp_bf16", autocast_dtype=torch.bfloat16, use_grad_scaler=False))
        else:
            print("Skipping BF16 training (unsupported).")
    else:
        print("CUDA not available; skipping training mixed precision benchmarks.")

    print("Training benchmarks:")
    for r in training_results:
        print(f"- {r['mode']}: {r['time_per_step_ms']:.2f} ms/step "
              f"(total {r['time_total_s']:.2f}s), peak mem {format_bytes(r['peak_memory'])}, "
              f"master params {format_bytes(r['param_master_bytes'])}")

    print()

    inference_results = []

    # FP32 inference (TF32 is still used internally for large matmuls on Ampere)
    inference_results.append(benchmark_inference(
        "infer_fp32(TF32 math)",
        build_fn=lambda: make_model().to(DEVICE),
        input_dtype=torch.float32,
        autocast_dtype=None  # not using autocast, just plain FP32 (TF32 math under the hood)
    ))

    # FP16 (convert weights)
    if DEVICE == "cuda":
        inference_results.append(benchmark_inference(
            "infer_fp16(weights_fp16)",
            build_fn=lambda: make_model().to(DEVICE).half(),
            input_dtype=torch.float16,
            autocast_dtype=None
        ))

        # BF16
        if torch.cuda.is_bf16_supported():
            inference_results.append(benchmark_inference(
                "infer_bf16(weights_bf16)",
                build_fn=lambda: make_model().to(DEVICE).bfloat16(),
                input_dtype=torch.bfloat16,
                autocast_dtype=None
            ))
        else:
            print("Skipping BF16 inference (unsupported).")

        # INT8 weight-only via bitsandbytes (optional)
        q8_model, err8 = try_bitsandbytes_linear_model("8")
        if q8_model is not None:
            # Inputs can be float16 for speed
            inference_results.append(benchmark_inference(
                "infer_int8(bitsandbytes)",
                build_fn=lambda m=q8_model: m,  # already built
                input_dtype=torch.float16,
                autocast_dtype=None
            ))
        else:
            print(f"INT8 bitsandbytes skipped: {err8}")

        # 4-bit NF4 weight-only (bitsandbytes)
        q4_model, err4 = try_bitsandbytes_linear_model("4")
        if q4_model is not None:
            inference_results.append(benchmark_inference(
                "infer_4bit_nf4(bitsandbytes)",
                build_fn=lambda m=q4_model: m,
                input_dtype=torch.float16,
                autocast_dtype=None
            ))
        else:
            print(f"4-bit bitsandbytes skipped: {err4}")

    else:
        print("CUDA not available; skipping GPU inference dtypes.")

    print("Inference benchmarks:")
    for r in inference_results:
        print(f"- {r['mode']}: {r['time_per_step_ms']:.2f} ms/step "
              f"(total {r['time_total_s']:.2f}s), peak mem {format_bytes(r['peak_memory'])}, "
              f"params {format_bytes(r['param_bytes'])}")

if __name__ == "__main__":
    main()