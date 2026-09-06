"""Apply CPU affinity, native thread limits and physical GPU restrictions early."""

from __future__ import annotations

import os
import random
import sys
import subprocess
from typing import Any


_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def configure_resources(config: Any, seed: int) -> dict[str, Any]:
    """Restrict this process and its children, then initialize deterministic RNGs.

    GPU IDs in config are physical IDs. An inherited numeric CUDA mask is
    intersected, never widened. Linux CPU affinity is a hard concurrency ceiling.
    """
    threads = config.cpu_threads
    if type(threads) is not int or not 1 <= threads <= 9:
        raise ValueError("cpu_threads must be an integer between 1 and 9")
    if not set(config.gpu_ids) <= {0, 1}:
        raise ValueError("only physical GPUs 0 and 1 are allowed")
    allowed = list(config.gpu_ids)
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None:
        tokens = [x.strip() for x in inherited.split(",") if x.strip()]
        if any(not x.isdigit() for x in tokens):
            if config.device != "cpu":
                raise ValueError("GPU execution requires an inherited numeric CUDA_VISIBLE_DEVICES mask")
            allowed = []
        else:
            allowed = [i for i in allowed if str(i) in tokens]
    if config.device == "cpu":
        visible = ""
        device = "cpu"
    else:
        physical = int(config.device.split(":")[1])
        if physical not in allowed:
            raise ValueError(f"physical GPU {physical} is excluded by the resource policy or inherited CUDA mask")
        visible = ",".join(map(str, allowed))
        device = f"cuda:{allowed.index(physical)}"
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        if os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
            raise RuntimeError("configure resources before CUDA initialization; start a new process")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for variable in _THREAD_VARS:
        os.environ[variable] = str(threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))[:threads]
        os.sched_setaffinity(0, affinity)
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    # Keep the controller alive: configured limits persist for this process.
    controller = threadpool_limits(limits=threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError("configure resources before starting parallel PyTorch work") from None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; select resources.device=cpu explicitly")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic
    gpu = None
    if device != "cpu":
        properties = torch.cuda.get_device_properties(device)
        gpu = {"name": properties.name, "compute_capability": [properties.major, properties.minor],
               "total_memory_bytes": properties.total_memory}
        try:
            driver = subprocess.run(
                ["nvidia-smi", "-i", str(physical), "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            gpu["uuid_driver"] = driver.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            gpu["uuid_driver"] = None
    return {
        "torch_version": torch.__version__, "cuda_build_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "gpu": gpu,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "device": device, "requested_device": config.device,
        "cuda_visible_devices": visible, "cpu_threads": threads,
        "cpu_affinity": affinity, "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "threadpool_controller": controller,
    }
