"""ScaleEvent-style MVSEC optical-flow metric core.

IMPORTANT: This is a *metric implementation*, not a reproduction of the entire
ScaleEvent MVSEC benchmark. Fair comparison additionally requires the same
sequence split, time intervals, GT alignment, resolution, event-mask geometry,
model supervision, and preprocessing.

The main-paper criterion is:
    valid = gt_valid AND at_least_one_event_in_flow_interval
    Outlier = (EPE > 3px) AND (EPE > 0.05 * ||GT||_2)

All errors use pixel displacement (NOT px/s) in the GT's pixel coordinate frame.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def event_activity_mask(
    events_xy_t_p: np.ndarray,
    start_s: float,
    end_s: float,
    height: int,
    width: int,
    *,
    timestamps_are_seconds: bool = True,
) -> np.ndarray:
    """Return [H,W] bool: pixel received >=1 event during (start,end].

    Expected events[N,4] columns x,y,t,polarity, with spatial coordinates
    already in EXACTLY the same image geometry as the GT and prediction.
    Does not rectify, resize, or crop for you. Must not be used on full streams
    copied into RAM indiscriminately; supply a sliced/small HDF5 event window.
    """
    e = np.asarray(events_xy_t_p)
    if e.ndim != 2 or e.shape[1] < 3:
        raise ValueError(f"expected events[N,>=3] x,y,t; got {e.shape}")
    if not (np.isfinite(start_s) and np.isfinite(end_s) and start_s < end_s):
        raise ValueError("invalid event interval")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if e.size == 0:
        return np.zeros((height, width), dtype=np.bool_)
    t = e[:, 2]
    if not timestamps_are_seconds:
        # e.g. microseconds in file, seconds in manifest
        t = t * 1.0e-6
    take = np.isfinite(t) & (t > start_s) & (t <= end_s)
    e = e[take]
    x = e[:, 0].astype(np.int64)
    y = e[:, 1].astype(np.int64)
    good = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    active = np.zeros((height, width), dtype=np.bool_)
    active[y[good], x[good]] = True
    return active


@dataclass
class ScaleEventFlowMeter:
    n_pixels: int = 0
    n_samples: int = 0
    sum_epe: float = 0.0
    n_out: int = 0
    n_1pe: int = 0
    n_2pe: int = 0
    n_3pe: int = 0
    # useful validity diagnostics
    n_gt_valid: int = 0
    n_valid_without_event: int = 0
    n_gt_mag_ge_60_and_3pe: int = 0
    n_3pe_not_outlier: int = 0

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        gt_valid: torch.Tensor,
        event_active: torch.Tensor,
    ) -> None:
        """Shapes: pred/gt=[B,2,H,W], masks=[B,H,W].

        Masks should be supplied in *native GT resolution*, with identical
        geometry and the SAME time window. We intentionally reject all shape
        mismatches instead of silently interpolating labels or flow vectors.
        """
        if pred.ndim != 4 or pred.shape[1] != 2 or pred.shape != gt.shape:
            raise ValueError(f"pred/gt mismatch: {tuple(pred.shape)}, {tuple(gt.shape)}")
        shape = (pred.shape[0], pred.shape[2], pred.shape[3])
        if gt_valid.shape != shape or event_active.shape != shape:
            raise ValueError(f"masks must be {shape}, got {gt_valid.shape}, {event_active.shape}")
        pred = pred.detach().float()
        gt = gt.detach().float()
        gt_valid = gt_valid.bool()
        event_active = event_active.bool()
        finite = torch.isfinite(pred).all(dim=1) & torch.isfinite(gt).all(dim=1)
        if ((~finite) & gt_valid & event_active).any().item():
            raise ValueError("non-finite prediction/GT on a valid active-event pixel")
        valid_gt = gt_valid & finite
        valid = valid_gt & event_active
        err = torch.linalg.vector_norm(pred - gt, dim=1)
        mag = torch.linalg.vector_norm(gt, dim=1)
        e = err[valid]
        g = mag[valid]
        self.n_samples += int(pred.shape[0])
        self.n_gt_valid += int(valid_gt.sum().item())
        self.n_valid_without_event += int((valid_gt & ~event_active).sum().item())
        if e.numel() == 0:
            return
        over3 = e > 3.0
        is_out = over3 & (e > .05 * g)
        self.n_pixels += int(e.numel())
        self.sum_epe += float(e.double().sum().item())
        self.n_out += int(is_out.sum().item())
        self.n_1pe += int((e > 1).sum().item())
        self.n_2pe += int((e > 2).sum().item())
        self.n_3pe += int(over3.sum().item())
        self.n_gt_mag_ge_60_and_3pe += int((over3 & (g >= 60)).sum().item())
        self.n_3pe_not_outlier += int((over3 & ~is_out).sum().item())

    def compute(self) -> dict[str, Any]:
        if not self.n_pixels:
            raise ValueError("no GT-valid event-active pixels; check event and GT alignment")
        n = self.n_pixels
        return {
            'EPE': self.sum_epe / n,
            'Outlier%': 100.0 * self.n_out / n,
            '1PE%': 100.0 * self.n_1pe / n,
            '2PE%': 100.0 * self.n_2pe / n,
            '3PE%': 100.0 * self.n_3pe / n,
            'valid_event_gt_pixels': n,
            'gt_valid_pixels': self.n_gt_valid,
            'gt_valid_without_event': self.n_valid_without_event,
            'event_conditioned_valid_fraction': n / self.n_gt_valid if self.n_gt_valid else None,
            '3PE_not_Outlier_pixel_count': self.n_3pe_not_outlier,
            'gt_magnitude_ge_60_and_3PE_pixel_count': self.n_gt_mag_ge_60_and_3pe,
            'samples_seen': self.n_samples,
        }
