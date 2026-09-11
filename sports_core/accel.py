"""
Inference backends
==================
PyTorch on CPU is the default, but on Intel hardware an OpenVINO export of the
same weights runs the models on the integrated GPU, which is dramatically
faster than the CPU.

Measured on an i3-1115G4 / Intel UHD iGPU, 720p input at imgsz=640:

    model                    PyTorch CPU    OpenVINO iGPU   speedup
    player_detector             1.11 s/f        0.23 s/f      4.9x
    ball_detector_model         1.16 s/f        0.21 s/f      5.6x
    court_keypoint_detector     2.37 s/f        0.37 s/f      6.3x
    ------------------------------------------------------------
    all three                   4.64 s/f        0.81 s/f      5.7x

Two findings worth knowing:

* OpenVINO on the **CPU** was *slower* than PyTorch here (~0.5x), so it is
  offered but never the default. The win comes from the iGPU.
* Exports use a **static batch of 1**. Dynamic-shape exports were ~4x slower
  on the iGPU (0.88 vs 0.20 s/frame), so callers must feed frames one at a
  time — `batch_size_for()` returns the right value.

Accuracy: player and court key-point outputs matched PyTorch exactly in
testing. The ball detector found slightly fewer low-confidence boxes (39 vs 44
over 8 frames); this was identical at FP32 and FP16, so it is a decode/NMS
difference rather than a precision loss.
"""

import os
import shutil
import tempfile
from pathlib import Path

CACHE_DIR = os.path.join(tempfile.gettempdir(), "sports_analysis_openvino")

PYTORCH_CPU = "pytorch_cpu"
PYTORCH_CUDA = "pytorch_cuda"
OPENVINO_GPU = "openvino_gpu"
OPENVINO_CPU = "openvino_cpu"

_LABELS = {
    PYTORCH_CPU: "PyTorch — CPU",
    PYTORCH_CUDA: "PyTorch — CUDA GPU",
    OPENVINO_GPU: "OpenVINO — Intel GPU (fastest here)",
    OPENVINO_CPU: "OpenVINO — CPU (usually slower)",
}

# Rough per-frame cost at imgsz=640, used only for the pre-run estimate.
SECONDS_PER_FRAME = {
    PYTORCH_CPU:  {"player": 1.1, "ball": 1.15, "court": 2.35},
    PYTORCH_CUDA: {"player": 0.05, "ball": 0.05, "court": 0.08},
    OPENVINO_GPU: {"player": 0.23, "ball": 0.21, "court": 0.40},
    OPENVINO_CPU: {"player": 2.8, "ball": 2.8, "court": 4.5},
}


# ── capability probing ───────────────────────────────────────────────────────

def openvino_installed():
    import importlib.util
    return importlib.util.find_spec("openvino") is not None


def openvino_devices():
    """Device names OpenVINO can see, e.g. ['CPU', 'GPU']. Empty if unavailable."""
    if not openvino_installed():
        return []
    try:
        from openvino import Core
        return list(Core().available_devices)
    except Exception:
        return []


def has_intel_gpu():
    return any(d.upper().startswith("GPU") for d in openvino_devices())


def available_backends():
    """(key, label) for every backend usable on this machine, best first."""
    import torch

    options = []
    if torch.cuda.is_available():
        options.append((PYTORCH_CUDA, _LABELS[PYTORCH_CUDA]))
    if has_intel_gpu():
        options.append((OPENVINO_GPU, _LABELS[OPENVINO_GPU]))
    options.append((PYTORCH_CPU, _LABELS[PYTORCH_CPU]))
    if openvino_installed():
        options.append((OPENVINO_CPU, _LABELS[OPENVINO_CPU]))
    return options


def default_backend():
    return available_backends()[0][0]


def label_for(backend):
    return _LABELS.get(backend, backend)


def is_openvino(backend):
    return backend in (OPENVINO_GPU, OPENVINO_CPU)


def predict_device(backend):
    """What to hand ultralytics as `device=`."""
    return {
        PYTORCH_CPU: "cpu",
        PYTORCH_CUDA: "cuda",
        OPENVINO_GPU: "intel:gpu",
        OPENVINO_CPU: "cpu",
    }[backend]


def batch_size_for(backend, default=20):
    """OpenVINO exports are static batch-1, so they must be fed one frame at a time."""
    return 1 if is_openvino(backend) else default


# ── export ───────────────────────────────────────────────────────────────────

def model_task(pt_path):
    """The task ('detect', 'pose', …) recorded in a .pt checkpoint.

    Must be passed explicitly when loading the exported IR: ultralytics cannot
    guess it from an OpenVINO directory and silently assumes 'detect', which
    turns the court key-point (pose) model into garbage.
    """
    from ultralytics import YOLO
    return YOLO(pt_path).task


def export_path(pt_path, imgsz, half=True):
    stem = Path(pt_path).stem
    tag = "fp16" if half else "fp32"
    return os.path.join(CACHE_DIR, f"{stem}_i{imgsz}_{tag}_openvino_model")


def is_exported(pt_path, imgsz, half=True):
    dest = export_path(pt_path, imgsz, half)
    return os.path.isdir(dest) and any(Path(dest).glob("*.xml"))


def export_openvino(pt_path, imgsz, half=True):
    """
    Convert a .pt to an OpenVINO IR directory, cached on disk.

    Returns (ir_directory, task). Raises on failure so the caller can fall
    back to PyTorch and say why.
    """
    from ultralytics import YOLO

    dest = export_path(pt_path, imgsz, half)
    task = model_task(pt_path)
    if is_exported(pt_path, imgsz, half):
        return dest, task

    os.makedirs(CACHE_DIR, exist_ok=True)

    kwargs = {"format": "openvino", "imgsz": imgsz, "verbose": False}
    if half:
        # 'half' was renamed 'quantize' in newer ultralytics; try the new name
        # first and fall back so both work.
        try:
            produced = YOLO(pt_path).export(quantize=16, **kwargs)
        except TypeError:
            produced = YOLO(pt_path).export(half=True, **kwargs)
    else:
        produced = YOLO(pt_path).export(**kwargs)

    # Ultralytics writes the IR next to the .pt; move it into our cache. The
    # directory name must keep its `_openvino_model` suffix to stay loadable.
    produced = str(produced).rstrip("\\/")
    if os.path.abspath(produced) != os.path.abspath(dest):
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(produced, dest)

    if not is_exported(pt_path, imgsz, half):
        raise RuntimeError(f"OpenVINO export produced no IR at {dest}")
    return dest, task


def resolve_model(pt_path, imgsz, backend, half=True):
    """
    Return (model_path, task) to hand to a detector class for this backend.

    For PyTorch backends this is just the .pt and task=None. For OpenVINO it
    exports on first use (slow once, cached thereafter).
    """
    if not is_openvino(backend):
        return pt_path, None
    return export_openvino(pt_path, imgsz, half=half)


def cache_size_mb():
    if not os.path.isdir(CACHE_DIR):
        return 0.0
    total = sum(f.stat().st_size for f in Path(CACHE_DIR).rglob("*") if f.is_file())
    return total / 1e6


def clear_cache():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
