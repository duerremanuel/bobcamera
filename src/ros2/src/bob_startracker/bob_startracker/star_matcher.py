"""Identify detected stars against the catalog and fit the fisheye model.

Because an all-sky camera's location and the capture time are known, this is
a much easier problem than blind plate solving: the only big unknowns are the
rotation of the camera about its axis (where north is in the image), whether
the image is mirrored and roughly the lens scale.  We

1. brute-force those (yaw 0..360, mirror, a few FOV scales) with a nominal
   equidistant model and count catalog stars that land near a detection,
2. iterate nearest-neighbour matching with a shrinking tolerance while
   fitting progressively more model parameters (rotation & scale ->
   principal point -> radial distortion -> tangential/aspect),
3. sigma-clip outliers.

Several captures (taken at different times, so the stars sit in different
places on the sensor) can be solved jointly because the camera does not move;
this gives far better coverage of the distortion near the horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .fisheye_model import FisheyeModel, azel_to_enu, angular_separation_deg, rotation_to_rvec
from .star_catalog import CatalogStar
from .star_detector import StarDetection


@dataclass
class StarMatch:
    det_index: int            # index into the capture's detections (-1 for manual)
    name: str
    mag: float
    az_deg: float
    el_deg: float
    x: float
    y: float
    kind: str = "star"
    manual: bool = False
    residual_px: float = float("nan")
    residual_deg: float = float("nan")

    def to_dict(self):
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class SolveCapture:
    """What the solver needs from a capture."""
    capture_id: str
    detections: List[StarDetection]
    stars: List[CatalogStar]                 # catalog positions at capture time
    manual: List[StarMatch] = field(default_factory=list)
    matches: List[StarMatch] = field(default_factory=list)   # output


@dataclass
class SolverParams:
    fov_deg: float = 180.0                 # nominal lens FOV across radius_px
    radius_px: Optional[float] = None      # image-circle radius; default short side / 2
    cx: Optional[float] = None
    cy: Optional[float] = None
    # lens scale search: multiplies fov_deg (i.e. divides the focal length)
    fov_scales: Sequence[float] = tuple(float(x) for x in np.geomspace(0.55, 1.6, 24))
    try_mirror: bool = True
    yaw_step_deg: float = 1.0
    coarse_min_el_deg: float = 25.0
    coarse_max_stars: int = 30
    coarse_max_dets: int = 80             # brightest detections used for the coarse search
    coarse_tol_frac: float = 0.05          # of radius
    min_el_deg: float = 5.0
    max_mag: float = 4.5
    final_tol_px: float = 3.0
    fit_tangential: bool = False
    fit_aspect: bool = False
    max_radial_terms: int = 4


@dataclass
class SolveResult:
    model: FisheyeModel
    n_matches: int
    rms_px: float
    rms_deg: float
    median_deg: float
    max_deg: float
    coarse_score: int
    fitted_params: List[str]
    per_capture: Dict[str, int]
    quality: str = "unknown"       # good | marginal | poor
    min_star_elevation_deg: float = float("nan")   # lowest calibration star

    def to_dict(self):
        d = asdict(self)
        d["model"] = self.model.to_dict()
        return d


# ---------------------------------------------------------------------------
# parameter packing
# ---------------------------------------------------------------------------

def _param_names(n_obs: int, sp: SolverParams, theta_cover_deg: float = 90.0) -> List[str]:
    """Free parameters for ``n_obs`` stars that reach ``theta_cover_deg`` off
    axis: higher radial terms only when there are enough stars AND they
    reach far enough out to constrain them."""
    names = ["r0", "r1", "r2", "f"]
    if n_obs >= 8:
        names += ["cx", "cy"]
    radial = 0
    for need, reach in ((12, 0.0), (25, 55.0), (45, 70.0), (80, 82.0)):
        if n_obs >= need and theta_cover_deg >= reach and radial < sp.max_radial_terms:
            radial += 1
    names += [f"k{i + 1}" for i in range(radial)]
    if sp.fit_tangential and n_obs >= 40:
        names += ["p1", "p2"]
    if sp.fit_aspect and n_obs >= 40:
        names += ["aspect"]
    return names


def _get(model: FisheyeModel, name: str) -> float:
    if name[0] == "r" and name[1:].isdigit():
        return model.rvec[int(name[1:])]
    if name[0] == "k":
        return model.k[int(name[1:]) - 1]
    if name[0] == "p" and name[1:].isdigit():
        return model.p[int(name[1:]) - 1]
    return getattr(model, name)


def _set(model: FisheyeModel, name: str, value: float):
    if name[0] == "r" and name[1:].isdigit():
        model.rvec[int(name[1:])] = value
    elif name[0] == "k":
        model.k[int(name[1:]) - 1] = value
    elif name[0] == "p" and name[1:].isdigit():
        model.p[int(name[1:]) - 1] = value
    else:
        setattr(model, name, value)


def _copy(model: FisheyeModel) -> FisheyeModel:
    return FisheyeModel.from_dict(model.to_dict())


def fit_model(model: FisheyeModel, xy: np.ndarray, azel: np.ndarray, names: List[str],
              f_scale: float = 2.0, weights: Optional[np.ndarray] = None,
              theta_limit_deg: Optional[float] = None) -> FisheyeModel:
    """Least-squares fit of ``names`` parameters to pixel observations.

    The radial polynomial is only trusted up to ``theta_limit_deg`` (default:
    a little beyond the outermost star) and is kept monotonic there, so the
    lens can never 'fold back' onto itself."""
    m = _copy(model)
    d_world = azel_to_enu(azel[:, 0], azel[:, 1])
    if theta_limit_deg is None:
        theta_limit_deg = min(_theta_cover(m, azel, robust=False) + 3.0, 110.0)
    m.theta_max_deg = theta_limit_deg
    t_grid = np.radians(np.linspace(1.0, theta_limit_deg, 24))
    w = np.ones(len(xy)) if weights is None else weights
    x0 = np.array([_get(m, n) for n in names], dtype=float)
    scales = np.array([m.f if n in ("f", "cx", "cy") else 1.0 for n in names])

    def resid(x):
        for n, val in zip(names, x * scales):
            _set(m, n, val)
        u, v, _ = m.world_to_pixel(d_world)
        # monotonic lens: r'(theta) must stay above 15% of f inside the limit
        mono = 50.0 * np.clip(0.15 - m._poly_dr(t_grid) / m.f, 0.0, None)
        return np.concatenate([(u - xy[:, 0]) * w, (v - xy[:, 1]) * w, mono])

    sol = least_squares(resid, x0 / scales, loss="soft_l1", f_scale=f_scale,
                        x_scale="jac", max_nfev=200 * (len(names) + 1))
    for n, val in zip(names, sol.x * scales):
        _set(m, n, val)
    return m


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def _det_array(dets: List[StarDetection]) -> np.ndarray:
    if not dets:
        return np.zeros((0, 2))
    return np.array([[d.x, d.y] for d in dets])


def match_capture(model: FisheyeModel, cap: SolveCapture, tol_px: float,
                  min_el: float, max_mag: float) -> List[StarMatch]:
    """Mutual-nearest-neighbour association within ``tol_px``."""
    stars = [s for s in cap.stars if s.el_deg >= min_el and s.mag <= max_mag]
    manual_names = {m.name.lower() for m in cap.manual}
    stars = [s for s in stars if s.name.lower() not in manual_names]
    det = _det_array(cap.detections)
    out = list(cap.manual)
    if not stars or len(det) == 0:
        return out
    u, v, vis = model.azel_to_pixel([s.az_deg for s in stars], [s.el_deg for s in stars])
    idx_vis = np.nonzero(vis)[0]
    if len(idx_vis) == 0:
        return out
    proj = np.stack([u[idx_vis], v[idx_vis]], axis=1)
    det_tree = cKDTree(det)
    proj_tree = cKDTree(proj)
    dist, di = det_tree.query(proj, distance_upper_bound=tol_px)
    for j, (dd, d_idx) in enumerate(zip(dist, di)):
        if not np.isfinite(dd):
            continue
        # mutual check: the closest projected star to that detection must be j
        _, back = proj_tree.query(det[d_idx])
        if back != j:
            continue
        s = stars[idx_vis[j]]
        out.append(StarMatch(int(d_idx), s.name, s.mag, s.az_deg, s.el_deg,
                             float(det[d_idx, 0]), float(det[d_idx, 1]), s.kind))
    return out


def coarse_search(cap: SolveCapture, width: int, height: int,
                  sp: SolverParams, n_candidates: int = 24) -> List[Tuple[FisheyeModel, int]]:
    """Brute-force yaw / mirror / scale.  Returns the best distinct
    candidates as (model, score), best first."""
    det = _det_array(cap.detections[: sp.coarse_max_dets])
    stars = [s for s in cap.stars if s.el_deg >= sp.coarse_min_el_deg][: sp.coarse_max_stars]
    if len(det) < 3 or len(stars) < 3:
        raise RuntimeError(f"not enough stars for coarse search "
                           f"({len(det)} detections, {len(stars)} catalog stars above "
                           f"{sp.coarse_min_el_deg} deg)")
    radius = sp.radius_px or min(width, height) / 2.0
    tol = sp.coarse_tol_frac * radius
    tree = cKDTree(det)
    d_world = azel_to_enu([s.az_deg for s in stars], [s.el_deg for s in stars])

    cands = []
    yaws = np.arange(0.0, 360.0, sp.yaw_step_deg)
    for mirror in ((False, True) if sp.try_mirror else (False,)):
        for fs in sp.fov_scales:
            base = FisheyeModel.nominal(width, height, sp.fov_deg * fs, 0.0, mirror,
                                        sp.cx, sp.cy, radius)
            u0, v0, _ = base.world_to_pixel(d_world)
            dx, dy = u0 - base.cx, v0 - base.cy
            for yaw in yaws:
                # rotating the zenith camera about its axis == rotating the image
                a = math.radians(yaw)
                ca, sa = math.cos(a), math.sin(a)
                if mirror:
                    sa = -sa
                px = base.cx + ca * dx - sa * dy
                py = base.cy + sa * dx + ca * dy
                dist, _ = tree.query(np.stack([px, py], 1), distance_upper_bound=tol)
                ok = np.isfinite(dist)
                score = int(ok.sum())
                cost = float(dist[ok].sum())
                cands.append((score, -cost, mirror, fs, yaw))
    cands.sort(reverse=True)
    out = []
    for score, _, mirror, fs, yaw in cands:
        # skip near-duplicates of an already selected candidate
        if any(m2 == mirror and abs(math.log(fs2 / fs)) < 0.1 and
               min(abs(y2 - yaw), 360 - abs(y2 - yaw)) < 6 for _, m2, fs2, y2 in out):
            continue
        out.append((score, mirror, fs, yaw))
        if len(out) >= n_candidates:
            break
    models = []
    for score, mirror, fs, yaw in out:
        model = FisheyeModel.nominal(width, height, sp.fov_deg * fs, 0.0, mirror, sp.cx, sp.cy, radius)
        # rotating the camera about its optical axis == rotating the image
        a = math.radians(yaw)
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        model.rvec = list(rotation_to_rvec(Rz @ model.R))
        models.append((model, score))
    return models


# ---------------------------------------------------------------------------
# full solve
# ---------------------------------------------------------------------------

def _stack(matches_by_cap: Dict[str, List[StarMatch]]):
    allm = [m for ms in matches_by_cap.values() for m in ms]
    xy = np.array([[m.x, m.y] for m in allm]) if allm else np.zeros((0, 2))
    azel = np.array([[m.az_deg, m.el_deg] for m in allm]) if allm else np.zeros((0, 2))
    return allm, xy, azel


def _theta_cover(model: FisheyeModel, azel: np.ndarray, robust: bool = True) -> float:
    """Off-axis angle (deg) reached by the matched stars (robust: the 2nd
    largest, so a single outlier cannot unlock extra distortion terms)."""
    if len(azel) == 0:
        return 0.0
    _, _, theta = model.world_to_pixel(azel_to_enu(azel[:, 0], azel[:, 1]))
    t = np.sort(np.degrees(theta))
    return float(t[-2] if robust and len(t) > 1 else t[-1])


def _refine(model: FisheyeModel, captures: List[SolveCapture], sp: SolverParams,
            tol: float, from_coarse: bool):
    min_el = max(sp.coarse_min_el_deg, sp.min_el_deg) if from_coarse else sp.min_el_deg
    names: List[str] = []
    rms = float("inf")
    for it in range(16):
        matches = {c.capture_id: match_capture(model, c, tol, min_el, sp.max_mag) for c in captures}
        allm, xy, azel = _stack(matches)
        if len(allm) < 4:
            return model, matches, float("inf"), names
        names = _param_names(len(allm), sp, _theta_cover(model, azel))
        if from_coarse and it == 0:
            names = ["r0", "r1", "r2", "f"]
        elif from_coarse and it == 1:
            names = [n for n in names if n[0] != "k" and n[0] != "p" and n != "aspect"]
        model = fit_model(model, xy, azel, names, f_scale=max(tol / 3.0, 1.0))
        u, v, _ = model.world_to_pixel(azel_to_enu(azel[:, 0], azel[:, 1]))
        res = np.hypot(u - xy[:, 0], v - xy[:, 1])
        rms = float(np.sqrt(np.mean(res ** 2)))
        if it >= 1:
            min_el = sp.min_el_deg          # model is roughly right: use the whole sky
        new_tol = max(sp.final_tol_px, min(tol, 5.0 * float(np.median(res)) + 1.0))
        if it >= 4 and abs(new_tol - tol) < 0.05:
            break
        tol = new_tol
    return model, matches, rms, names


def solve(captures: List[SolveCapture], width: int, height: int,
          sp: Optional[SolverParams] = None,
          initial: Optional[FisheyeModel] = None) -> SolveResult:
    sp = sp or SolverParams()
    if not captures:
        raise RuntimeError("no captures to solve")
    radius = sp.radius_px or min(width, height) / 2.0

    score = -1
    if initial is None:
        ref = max(captures, key=lambda c: min(len(c.detections), 3 * len(c.stars)))
        best = None
        for cand, cscore in coarse_search(ref, width, height, sp):
            m, matches, rms, names = _refine(cand, captures, sp, sp.coarse_tol_frac * radius, True)
            # score by how many stars fit *tightly*: a wrong solution can
            # still collect loose matches by bending the distortion terms
            allm, xy, azel = _stack(matches)
            if len(allm) == 0:
                continue
            u, v, _ = m.world_to_pixel(azel_to_enu(azel[:, 0], azel[:, 1]))
            res = np.hypot(u - xy[:, 0], v - xy[:, 1])
            n_tight = int((res < 2.0 * sp.final_tol_px).sum())
            key = (n_tight, -rms)
            if best is None or key > best[0]:
                best = (key, m, cscore)
        if best is None or best[0][0] < 4:
            raise RuntimeError("could not identify the star field - check site location, "
                               "capture time (UTC), mask and fov settings")
        model, score = best[1], best[2]
    else:
        model = _copy(initial)
        model.theta_max_deg = None
    model, matches, rms, names = _refine(model, captures, sp, max(4 * sp.final_tol_px, 8.0), False)
    tol = max(sp.final_tol_px, 4.0 * rms)
    matches = {c.capture_id: match_capture(model, c, tol, sp.min_el_deg, sp.max_mag) for c in captures}
    allm, _, _ = _stack(matches)
    if len(allm) < 4:
        raise RuntimeError(f"only {len(allm)} stars matched - check site location, "
                           f"capture time (UTC), mask and fov settings")

    # sigma-clip and refit
    for _ in range(4):
        allm, xy, azel = _stack(matches)
        names = _param_names(len(allm), sp, _theta_cover(model, azel))
        model = fit_model(model, xy, azel, names, f_scale=1.0)
        u, v, _ = model.world_to_pixel(azel_to_enu(azel[:, 0], azel[:, 1]))
        res = np.hypot(u - xy[:, 0], v - xy[:, 1])
        rms = float(np.sqrt(np.mean(res ** 2)))
        clip = max(3.0 * rms, 1.0)
        keep_ids = {id(m) for m, r in zip(allm, res) if r <= clip or m.manual}
        if len(keep_ids) == len(allm):
            break
        matches = {k: [m for m in ms if id(m) in keep_ids] for k, ms in matches.items()}

    allm, xy, azel = _stack(matches)
    _, _, theta = model.world_to_pixel(azel_to_enu(azel[:, 0], azel[:, 1]))
    model.theta_max_deg = float(np.degrees(theta.max())) + 2.0
    u, v, _ = model.world_to_pixel(azel_to_enu(azel[:, 0], azel[:, 1]))
    res_px = np.hypot(u - xy[:, 0], v - xy[:, 1])
    az_f, el_f = model.pixel_to_azel(xy[:, 0], xy[:, 1])
    res_deg = angular_separation_deg(az_f, el_f, azel[:, 0], azel[:, 1])
    for m, rp, rd in zip(allm, res_px, res_deg):
        m.residual_px = float(rp)
        m.residual_deg = float(rd)
    for c in captures:
        c.matches = matches[c.capture_id]

    return SolveResult(
        model=model,
        n_matches=len(allm),
        rms_px=float(np.sqrt(np.mean(res_px ** 2))),
        rms_deg=float(np.sqrt(np.mean(res_deg ** 2))),
        median_deg=float(np.median(res_deg)),
        max_deg=float(np.max(res_deg)),
        coarse_score=score,
        fitted_params=names,
        per_capture={k: len(v) for k, v in matches.items()},
        min_star_elevation_deg=float(azel[:, 1].min()),
        quality=_quality(len(allm), float(np.sqrt(np.mean(res_px ** 2))), sp),
    )


def _quality(n: int, rms_px: float, sp: SolverParams) -> str:
    if n >= 15 and rms_px <= max(2.0, sp.final_tol_px * 0.7):
        return "good"
    if n >= 8 and rms_px <= max(4.0, sp.final_tol_px * 1.5):
        return "marginal"
    return "poor"
