"""Helpers shared by the state adapters (numpy/torch only)."""
import os

import numpy as np

AMP = os.environ.get("TSFM_AMP", "fp16")      # "fp16" autocast (RTX 2070 has no bf16) or "fp32"
CHUNK = int(os.environ.get("TSFM_CHUNK", "32"))


def autocast(torch, device):
    if AMP == "fp32" or not str(device).startswith("cuda"):
        import contextlib
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.float16)


def chunks(n, size=None):
    size = size or CHUNK
    for i in range(0, n, size):
        yield slice(i, min(n, i + size))


class Capture:
    """Forward (pre-)hook that keeps the last tensor seen by a module."""

    def __init__(self, module, pre=True, index=0):
        self.x = None
        self.index = index
        if pre:
            self.h = module.register_forward_pre_hook(self._pre)
        else:
            self.h = module.register_forward_hook(self._post)

    def _pre(self, mod, args, kwargs=None):
        self.x = args[self.index]

    def _post(self, mod, args, out):
        self.x = out[self.index] if isinstance(out, (tuple, list)) else out
