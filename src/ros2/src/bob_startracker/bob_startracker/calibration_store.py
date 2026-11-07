"""On-disk storage of star captures and the resulting calibration.

Layout (default root ``assets/startracker``)::

    captures/<capture_id>/image.png      stacked frame used for detection
    captures/<capture_id>/capture.json   time, site, detections, star matches
    captures/<capture_id>/overlay.jpg    detections (red) / matched stars (green)
    calibration.json                     current fisheye -> az/el model + stats
    calibration_history/<stamp>.json     previous calibrations

Everything is plain JSON so other tools (web UI, digital twin, PTZ slew
controllers) can read the calibration without importing this package.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import cv2
import numpy as np

from .fisheye_model import FisheyeModel
from .star_catalog import Site, StarCatalog
from .star_detector import StarDetection, DetectorParams, detect_stars
from .star_matcher import StarMatch, SolveCapture, SolverParams, SolveResult, solve, match_capture


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


@dataclass
class Capture:
    capture_id: str
    utc_time: datetime
    site: Site
    image_width: int
    image_height: int
    detections: List[StarDetection] = field(default_factory=list)
    matches: List[StarMatch] = field(default_factory=list)
    manual: List[StarMatch] = field(default_factory=list)
    frames_stacked: int = 1
    sun_elevation_deg: float = float("nan")
    notes: str = ""
    conditions: Optional[dict] = None     # weather / camera cloud estimate at capture time

    def to_dict(self):
        return {
            "capture_id": self.capture_id,
            "utc_time": self.utc_time.isoformat(),
            "site": self.site.to_dict(),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "frames_stacked": self.frames_stacked,
            "sun_elevation_deg": self.sun_elevation_deg,
            "notes": self.notes,
            "conditions": self.conditions,
            "detections": [d.to_dict() for d in self.detections],
            "matches": [m.to_dict() for m in self.matches],
            "manual": [m.to_dict() for m in self.manual],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            capture_id=d["capture_id"],
            utc_time=parse_time(d["utc_time"]),
            site=Site.from_dict(d["site"]),
            image_width=int(d["image_width"]),
            image_height=int(d["image_height"]),
            detections=[StarDetection(**x) for x in d.get("detections", [])],
            matches=[StarMatch.from_dict(x) for x in d.get("matches", [])],
            manual=[StarMatch.from_dict(x) for x in d.get("manual", [])],
            frames_stacked=int(d.get("frames_stacked", 1)),
            sun_elevation_deg=float(d.get("sun_elevation_deg", float("nan"))),
            notes=d.get("notes", ""),
            conditions=d.get("conditions"),
        )


class CalibrationStore:

    def __init__(self, root: str = "assets/startracker", catalog: Optional[StarCatalog] = None):
        self.root = root
        self.captures_dir = os.path.join(root, "captures")
        os.makedirs(self.captures_dir, exist_ok=True)
        self.catalog = catalog or StarCatalog()

    # ----- paths -------------------------------------------------------------
    def capture_dir(self, cid: str) -> str:
        return os.path.join(self.captures_dir, cid)

    @property
    def calibration_path(self) -> str:
        return os.path.join(self.root, "calibration.json")

    # ----- captures ------------------------------------------------------------
    def list_captures(self) -> List[str]:
        if not os.path.isdir(self.captures_dir):
            return []
        return sorted(d for d in os.listdir(self.captures_dir)
                      if os.path.isfile(os.path.join(self.captures_dir, d, "capture.json")))

    def load_capture(self, cid: str) -> Capture:
        with open(os.path.join(self.capture_dir(cid), "capture.json")) as fh:
            return Capture.from_dict(json.load(fh))

    def load_image(self, cid: str) -> np.ndarray:
        return cv2.imread(os.path.join(self.capture_dir(cid), "image.png"), cv2.IMREAD_UNCHANGED)

    def save_capture(self, cap: Capture, image: Optional[np.ndarray] = None):
        d = self.capture_dir(cap.capture_id)
        os.makedirs(d, exist_ok=True)
        if image is not None:
            img = image
            if img.dtype != np.uint8 and img.dtype != np.uint16:
                img = np.clip(img, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(d, "image.png"), img)
        tmp = os.path.join(d, "capture.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(cap.to_dict(), fh, indent=1)
        os.replace(tmp, os.path.join(d, "capture.json"))

    def delete_capture(self, cid: str):
        shutil.rmtree(self.capture_dir(cid), ignore_errors=True)

    def new_capture(self, image: np.ndarray, utc_time: datetime, site: Site,
                    mask: Optional[np.ndarray] = None,
                    detector: Optional[DetectorParams] = None,
                    frames_stacked: int = 1, notes: str = "",
                    conditions: Optional[dict] = None) -> Capture:
        """Detect stars in ``image`` and store it as a new capture."""
        h, w = image.shape[:2]
        cid = utc_time.strftime("%Y%m%dT%H%M%SZ")
        base, n = cid, 1
        while os.path.exists(self.capture_dir(cid)):
            cid = f"{base}_{n}"
            n += 1
        dets = detect_stars(image, mask, detector)
        cap = Capture(cid, utc_time, site, w, h, dets, frames_stacked=frames_stacked,
                      sun_elevation_deg=StarCatalog.sun_elevation(site, utc_time), notes=notes,
                      conditions=conditions)
        self.save_capture(cap, image)
        return cap

    def add_manual_star(self, cid: str, name: str, x: float, y: float) -> StarMatch:
        """Pin a catalog star to a pixel position by hand (always used in fits)."""
        cap = self.load_capture(cid)
        s = self.catalog.position(name, cap.site, cap.utc_time)
        if s is None:
            raise KeyError(f"'{name}' is not in the catalog")
        m = StarMatch(-1, s.name, s.mag, s.az_deg, s.el_deg, float(x), float(y), s.kind, manual=True)
        cap.manual = [mm for mm in cap.manual if mm.name.lower() != s.name.lower()] + [m]
        self.save_capture(cap)
        return m

    # ----- solving ---------------------------------------------------------------
    def _solve_capture(self, cap: Capture, max_mag: float) -> SolveCapture:
        stars = self.catalog.visible(cap.site, cap.utc_time, min_elevation_deg=0.0, max_mag=max_mag)
        return SolveCapture(cap.capture_id, cap.detections, stars, list(cap.manual))

    def calibrate(self, capture_ids: Optional[List[str]] = None,
                  sp: Optional[SolverParams] = None,
                  use_existing: bool = False, force: bool = False) -> SolveResult:
        """Solve all (or the given) captures jointly and write calibration.json.

        A 'poor' solution is returned but NOT saved unless ``force``."""
        sp = sp or SolverParams()
        ids = capture_ids or self.list_captures()
        if not ids:
            raise RuntimeError("no captures stored yet")
        caps = [self.load_capture(c) for c in ids]
        w, h = caps[0].image_width, caps[0].image_height
        for c in caps:
            if (c.image_width, c.image_height) != (w, h):
                raise RuntimeError(f"capture {c.capture_id} has a different resolution "
                                   f"({c.image_width}x{c.image_height} vs {w}x{h})")
        initial = self.load_model() if use_existing and os.path.exists(self.calibration_path) else None
        if initial is not None and (initial.image_width, initial.image_height) != (w, h):
            initial = initial.scaled(w, h)

        solve_caps = [self._solve_capture(c, sp.max_mag) for c in caps]
        if initial is None and len(solve_caps) > 1:
            # bootstrap on the best single capture, then solve jointly
            ref = max(solve_caps, key=lambda c: min(len(c.detections), 3 * len(c.stars)))
            initial = solve([ref], w, h, sp).model
        result = solve(solve_caps, w, h, sp, initial)
        if result.quality == "poor" and not force:
            raise RuntimeError(f"calibration rejected as poor ({result.n_matches} stars, "
                               f"rms {result.rms_px:.1f} px) - calibration.json not changed; "
                               f"check site/time/mask or add manual stars (force to save anyway)")

        for cap, sc in zip(caps, solve_caps):
            cap.matches = [m for m in sc.matches if not m.manual]
            for m in cap.manual:
                hit = next((x for x in sc.matches if x.manual and x.name == m.name), None)
                if hit:
                    m.residual_px, m.residual_deg = hit.residual_px, hit.residual_deg
            self.save_capture(cap)
            self.write_overlay(cap, result.model)
        self.save_calibration(result, ids)
        return result

    # ----- calibration file ---------------------------------------------------------
    def save_calibration(self, result: SolveResult, capture_ids: List[str]):
        doc = {
            "created_utc": utcnow().isoformat(),
            "captures": capture_ids,
            "model": result.model.to_dict(),
            "summary": result.model.describe(),
            "fit": {
                "n_stars": result.n_matches,
                "rms_px": result.rms_px,
                "rms_deg": result.rms_deg,
                "median_deg": result.median_deg,
                "max_deg": result.max_deg,
                "fitted_params": result.fitted_params,
                "per_capture": result.per_capture,
                "quality": result.quality,
                # below this the mapping is extrapolated (no calibration stars)
                "min_star_elevation_deg": result.min_star_elevation_deg,
                "theta_max_deg": result.model.theta_max_deg,
            },
        }
        if os.path.exists(self.calibration_path):
            hist = os.path.join(self.root, "calibration_history")
            os.makedirs(hist, exist_ok=True)
            shutil.copy(self.calibration_path,
                        os.path.join(hist, utcnow().strftime("%Y%m%dT%H%M%SZ") + ".json"))
        tmp = self.calibration_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=2)
        os.replace(tmp, self.calibration_path)
        return doc

    def load_calibration(self) -> dict:
        with open(self.calibration_path) as fh:
            return json.load(fh)

    def load_model(self) -> FisheyeModel:
        return FisheyeModel.from_dict(self.load_calibration()["model"])

    # ----- visual check ----------------------------------------------------------------
    def write_overlay(self, cap: Capture, model: Optional[FisheyeModel] = None,
                      path: Optional[str] = None) -> Optional[str]:
        img = self.load_image(cap.capture_id)
        if img is None:
            return None
        if img.dtype != np.uint8:
            img = cv2.convertScaleAbs(img, alpha=255.0 / max(float(img.max()), 1.0))
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # stretch so faint stars are visible
        img = cv2.convertScaleAbs(img, alpha=3.0)
        for d in cap.detections:
            cv2.circle(img, (int(round(d.x)), int(round(d.y))), 6, (0, 0, 255), 1)
        if model is not None:
            draw_azel_grid(img, model)
            stars = self.catalog.visible(cap.site, cap.utc_time, 0.0, 4.5)
            u, v, vis = model.azel_to_pixel([s.az_deg for s in stars], [s.el_deg for s in stars])
            for s, uu, vv, ok in zip(stars, u, v, vis):
                if ok:
                    cv2.drawMarker(img, (int(round(uu)), int(round(vv))), (255, 200, 0),
                                   cv2.MARKER_CROSS, 8, 1)
        for m in cap.matches + cap.manual:
            c = (0, 255, 0) if not m.manual else (0, 255, 255)
            cv2.circle(img, (int(round(m.x)), int(round(m.y))), 10, c, 1)
            cv2.putText(img, m.name, (int(m.x) + 12, int(m.y) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, c, 1, cv2.LINE_AA)
        path = path or os.path.join(self.capture_dir(cap.capture_id), "overlay.jpg")
        cv2.imwrite(path, img)
        return path


def draw_azel_grid(img: np.ndarray, model: FisheyeModel, color=(120, 120, 120)):
    """Draw elevation circles every 15 deg and azimuth spokes every 30 deg,
    with N/E/S/W labels at the horizon."""
    def poly(az, el):
        u, v, vis = model.azel_to_pixel(az, el)
        pts = np.stack([u, v], 1)[vis]
        return pts.astype(np.int32)

    for el in (0, 15, 30, 45, 60, 75):
        az = np.linspace(0, 360, 361)
        pts = poly(az, np.full_like(az, el))
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
    for az in range(0, 360, 30):
        el = np.linspace(0, 90, 91)
        pts = poly(np.full_like(el, az), el)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
    for label, az in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        u, v, vis = model.azel_to_pixel([az], [3.0])
        if vis[0]:
            cv2.putText(img, label, (int(u[0]) - 6, int(v[0]) + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 200, 255), 2, cv2.LINE_AA)
