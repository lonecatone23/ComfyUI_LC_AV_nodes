import logging
from typing import Callable, Optional

import torch

log = logging.getLogger()


# Partially from https://github.com/gle-bellier/flow-matching
# LC_AV: euler inference only (the 'adaptive' mode needed torchdiffeq), training helpers removed, and a per-step
# callback in place of the ComfyUI/tqdm progress bars, so the caller can draw progress and stop on Cancel.
class FlowMatching:

    def __init__(self, min_sigma: float = 0.0, inference_mode='euler', num_steps: int = 25):
        super().__init__()
        self.min_sigma = min_sigma
        self.inference_mode = inference_mode
        self.num_steps = num_steps
        assert self.inference_mode == 'euler', 'this vendored copy only has the euler sampler'

    def to_data(self, fn: Callable, x0: torch.Tensor, callback: Optional[Callable] = None) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x0, 0, 1, callback)

    def run_t0_to_t1(self, fn: Callable, x0: torch.Tensor, t0: float, t1: float,
                     callback: Optional[Callable] = None) -> torch.Tensor:
        # fn: a function that takes (t, x) and returns the direction x0->x1
        x = x0
        steps = torch.linspace(t0, t1 - self.min_sigma, self.num_steps + 1)
        for ti, t in enumerate(steps[:-1]):
            flow = fn(t, x)
            next_t = steps[ti + 1]
            dt = next_t - t
            x = x + dt * flow
            if callback is not None:
                callback(ti + 1, self.num_steps)
        return x
