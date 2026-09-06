"""Resource restrictions are checked in fresh processes to avoid CUDA state leaks."""

import json
import os
import subprocess
import sys

import pytest


def resource_process(config, mask="0,1"):
    code = f'''
import json
from active_learning_docking.config import ResourceConfig
from active_learning_docking.resources import configure_resources
config = ResourceConfig(**json.loads({json.dumps(json.dumps(config))}))
result = configure_resources(config, 42)
result.pop("threadpool_controller")
print(json.dumps(result))
'''
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=mask)
    return subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)


def test_cpu_policy_and_thread_limits():
    result = resource_process({"cpu_threads": 1, "device": "cpu"})
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info["torch_threads"] == info["torch_interop_threads"] == 1
    assert len(info["cpu_affinity"]) == 1
    assert info["cuda_visible_devices"] == ""
    assert info["deterministic_algorithms"]


@pytest.mark.parametrize("threads", [0, 10, 64, True])
def test_unsafe_cpu_limit_rejected(threads):
    result = resource_process({"cpu_threads": threads})
    assert result.returncode != 0
    assert "cpu_threads" in result.stderr


def test_excluded_physical_gpu_cannot_be_reenabled():
    result = resource_process({"device": "cuda:1"}, mask="0")
    assert result.returncode != 0
    assert "excluded" in result.stderr


def test_third_gpu_rejected():
    result = resource_process({"gpu_ids": [2], "device": "cpu"})
    assert result.returncode != 0
    assert "only physical GPUs" in result.stderr
