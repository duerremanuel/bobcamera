"""Detect star-like point sources in an (all-sky) night frame with sub-pixel
centroids.

Pipeline: grey -> optional stack/mean -> subtract smooth background (median
filter) -> robust noise (MAD) -> threshold at ``snr`` sigma -> connected
components -> reject blobs that are too big / elongated (lights, moon, planes,
clouds edges) -> intensity weighted centroid on the background-subtracted
image.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class StarDetection:
    x: float
    y: float
    flux: float
    peak: float
    snr: float
    area: int
    fwhm: float

    def to_dict(self):
        return {k: float(v) if not isinstance(v, int) else v for k, v in asdict(self).items()}


@dataclass
class DetectorParams:
    snr: float = 5.0                 # detection threshold in noise sigmas
    background_kernel: int = 31      # median filter size for background
    min_area: int = 1
    max_area: int = 150              # px at 1080p (scaled with resolution); larger blobs are not stars
    max_elongation: float = 3.0      # ratio of principal axes
    max_stars: int = 400
    border: int = 4                  # ignore detections this close to the edge
    saturation: float = 254.0        # informational; saturated stars are kept


def to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32)


def stack_frames(frames: List[np.ndarray]) -> np.ndarray:
    """Mean-stack frames (reduces noise by sqrt(N); stars move ~0.25 deg/min
    so keep the stack span to a few seconds)."""
    acc = None
    for f in frames:
        g = to_gray_float(f)
        acc = g if acc is None else acc + g
    return acc / float(len(frames))


def _robust_stats(x: np.ndarray):
    """Median and noise sigma: MAD, falling back to a sigma-clipped std when
    the data are so quantised that the MAD collapses."""
    med = float(np.median(x))
    mad = float(1.4826 * np.median(np.abs(x - med)))
    y = x
    s = float(y.std())
    for _ in range(5):
        y = y[np.abs(y - med) < 3.0 * max(s, 1e-6)]
        if y.size < 10:
            break
        s = float(y.std())
    return med, max(mad, s, 0.3)


def detect_stars(image: np.ndarray, mask: Optional[np.ndarray] = None,
                 params: Optional[DetectorParams] = None) -> List[StarDetection]:
    p = params or DetectorParams()
    gray = to_gray_float(image)
    h, w = gray.shape

    k = p.background_kernel | 1
    # median blur on 8-bit is fast and supports big kernels; scale to keep precision
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    if gray.max() > 255:
        scale = 255.0 / float(gray.max())
        g8 = np.clip(gray * scale, 0, 255).astype(np.uint8)
        background = cv2.medianBlur(g8, k).astype(np.float32) / scale
    else:
        background = cv2.medianBlur(g8, k).astype(np.float32)
    resid = gray - background

    valid = np.ones((h, w), dtype=bool)
    if mask is not None:
        m = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        valid &= m > 127
    # ignore the black area outside the fisheye image circle and a margin
    # inside it: the sharp rim produces lots of spurious "stars"
    if valid.any():
        bg_level = float(np.median(background[valid]))
        dark = (background < max(3.0, 0.3 * bg_level)).astype(np.uint8)
        if dark.any():
            margin = max(5, k // 3)
            dark = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2))
            valid &= dark == 0
    b = p.border
    valid[:b, :] = valid[-b:, :] = False
    valid[:, :b] = valid[:, -b:] = False

    # Measure the noise on the *smoothed* image that is thresholded: real
    # frames are often compressed/upscaled, so their noise is spatially
    # correlated and the raw per-pixel noise underestimates it.
    smooth = cv2.GaussianBlur(resid, (3, 3), 0.8)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return []
    step = max(1, n_valid // 400_000)
    med, sigma = _robust_stats(smooth[valid][::step])
    sigma_raw = _robust_stats(resid[valid][::step])[1]
    # big sensors have proportionally bigger star images
    max_area = max(p.max_area, int(p.max_area * (min(h, w) / 1080.0) ** 2))

    binary = ((smooth - med) > p.snr * sigma) & valid
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)

    out: List[StarDetection] = []
    for i in range(1, n):
        x0, y0, bw, bh, area = stats[i]
        if area < p.min_area or area > max_area:
            continue
        pad = 2
        xa, ya = max(x0 - pad, 0), max(y0 - pad, 0)
        xb, yb = min(x0 + bw + pad, w), min(y0 + bh + pad, h)
        patch = resid[ya:yb, xa:xb] - med
        lab = labels[ya:yb, xa:xb]
        sel = cv2.dilate((lab == i).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        wts = np.where(sel, np.clip(patch, 0, None), 0.0)
        flux = float(wts.sum())
        if flux <= 0:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        cxs = float((wts * xx).sum() / flux)
        cys = float((wts * yy).sum() / flux)
        # second moments -> elongation and fwhm
        mxx = float((wts * (xx - cxs) ** 2).sum() / flux)
        myy = float((wts * (yy - cys) ** 2).sum() / flux)
        mxy = float((wts * (xx - cxs) * (yy - cys)).sum() / flux)
        tr, det = mxx + myy, mxx * myy - mxy * mxy
        disc = max(tr * tr / 4 - det, 0.0) ** 0.5
        l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
        if (l1 / l2) ** 0.5 > p.max_elongation:
            continue
        fwhm = 2.3548 * ((l1 + l2) / 2) ** 0.5
        peak = float(patch[sel].max())
        noise = sigma_raw * max(float(sel.sum()), 1.0) ** 0.5
        out.append(StarDetection(cxs, cys, flux, peak, flux / noise, int(area), fwhm))

    out.sort(key=lambda s: -s.flux)
    return out[: p.max_stars]


def find_image_circle(image: np.ndarray, threshold: Optional[float] = None):
    """Estimate the fisheye image circle (cx, cy, radius) from the region that
    is not black.  Works on daytime / twilight frames; returns None if the
    frame does not show a clear circle."""
    gray = to_gray_float(image)
    if threshold is None:
        threshold = max(8.0, float(np.percentile(gray, 5)) + 6.0)
    m = (cv2.GaussianBlur(gray, (5, 5), 0) > threshold).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    h, w = gray.shape
    # a circle that is clipped by the frame on all sides isn't informative
    if r > 0.75 * np.hypot(w, h) / 2:
        return None
    return float(x), float(y), float(r)
