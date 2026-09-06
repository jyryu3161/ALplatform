"""Keep the test runner and child processes within the project's CPU policy."""

import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])


def pytest_sessionstart(session):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
