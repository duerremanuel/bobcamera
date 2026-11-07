"""Fisheye camera model mapping image pixels <-> (true-north azimuth, elevation).

Conventions
-----------
* World frame is local ENU (x = East, y = North, z = Up).
* Azimuth is measured from TRUE (geographic) north, clockwise through east, in
  degrees [0, 360).  Elevation is degrees above the horizon (refraction is
  already included in the catalog positions, so the model maps to *apparent*
  elevation, which is what the camera sees).
* Camera frame follows the usual pinhole convention: x = image right (+u),
  y = image down (+v), z = optical axis (out of the lens).  For an all-sky
  camera pointing at the zenith z ~ Up.  Because this frame is right handed,
  an upward-looking camera with north at the top of the image naturally has
  east on the LEFT, as on a star chart.  ``mirror`` handles cameras/streams
  that are flipped horizontally on top of that.

Projection (Kannala-Brandt / OpenCV-fisheye style radial model)
--------------------------------------------------------------
    theta = angle between the ray and the optical axis
    r(theta) = f * (theta + k1 theta^3 + k2 theta^5 + k3 theta^7 + k4 theta^9)
    u = cx + r cos(phi)            (negated when mirror is set)
    v = cy + aspect * r sin(phi)

plus optional tangential (decentering) terms p1, p2 applied in normalised
coordinates.  Equidistant, equisolid, stereographic and orthographic lenses
are all well approximated by the odd polynomial within a 180-190 deg FOV.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# rotation helpers
# ---------------------------------------------------------------------------

def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation vector -> 3x3 rotation matrix."""
    rvec = np.asarray(rvec, dtype=float)
    angle = np.linalg.norm(rvec)
    if angle < 1e-15:
        return np.eye(3)
    k = rvec / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def rotation_to_rvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> rotation vector (robust near 0 and pi)."""
    R = np.asarray(R, dtype=float)
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(cos_a)
    if angle < 1e-12:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        # near 180 deg: axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(M), 0, None))
        i = int(np.argmax(axis))
        axis[i] = math.sqrt(M[i, i])
        for j in range(3):
            if j != i:
                axis[j] = M[i, j] / axis[i]
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * math.sin(angle)) * angle


def zenith_rotation(up_azimuth_deg: float) -> np.ndarray:
    """World->camera rotation for a camera looking straight up whose image
    'up' direction (-v) points towards ``up_azimuth_deg`` on the horizon."""
    psi = math.radians(up_azimuth_deg)
    h = np.array([math.sin(psi), math.cos(psi), 0.0])   # horizontal dir of image-up
    z_cam = np.array([0.0, 0.0, 1.0])
    y_cam = -h
    x_cam = np.cross(y_cam, z_cam)
    return np.vstack([x_cam, y_cam, z_cam])


# ---------------------------------------------------------------------------
# az/el <-> unit vectors
# ---------------------------------------------------------------------------

def azel_to_enu(az_deg, el_deg) -> np.ndarray:
    az = np.radians(np.asarray(az_deg, dtype=float))
    el = np.radians(np.asarray(el_deg, dtype=float))
    ce = np.cos(el)
    return np.stack([ce * np.sin(az), ce * np.cos(az), np.sin(el)], axis=-1)


def enu_to_azel(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    az = np.degrees(np.arctan2(v[..., 0], v[..., 1])) % 360.0
    el = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    return az, el


def angular_separation_deg(az1, el1, az2, el2) -> np.ndarray:
    a = azel_to_enu(az1, el1)
    b = azel_to_enu(az2, el2)
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.degrees(np.arctan2(cross, dot))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

@dataclass
class FisheyeModel:
    image_width: int
    image_height: int
    cx: float
    cy: float
    f: float                      # pixels per radian at the centre
    aspect: float = 1.0           # fy / fx
    k: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])   # k1..k4
    p: list = field(default_factory=lambda: [0.0, 0.0])             # p1, p2 tangential
    rvec: list = field(default_factory=lambda: [0.0, 0.0, 0.0])     # world(ENU)->camera
    mirror: bool = False
    # Largest off-axis angle (deg) supported by calibration stars.  Beyond it
    # the radial law is continued quadratically (matching r, r', r'') instead
    # of trusting the polynomial, which can swing wildly outside the fitted
    # range (e.g. below trees that hide the horizon).
    theta_max_deg: Optional[float] = None

    # ----- construction ---------------------------------------------------
    @classmethod
    def nominal(cls, width: int, height: int, fov_deg: float = 180.0,
                up_azimuth_deg: float = 0.0, mirror: bool = False,
                cx: float | None = None, cy: float | None = None,
                radius_px: float | None = None) -> "FisheyeModel":
        """Equidistant zenith-pointing model whose image circle (radius_px,
        default half the short side) spans ``fov_deg``."""
        if radius_px is None:
            radius_px = min(width, height) / 2.0
        f = radius_px / math.radians(fov_deg / 2.0)
        R = zenith_rotation(up_azimuth_deg)
        return cls(width, height,
                   cx if cx is not None else (width - 1) / 2.0,
                   cy if cy is not None else (height - 1) / 2.0,
                   f, rvec=list(rotation_to_rvec(R)), mirror=mirror)

    @property
    def R(self) -> np.ndarray:
        return rodrigues(np.asarray(self.rvec))

    # ----- radial polynomial ------------------------------------------------
    def _poly_r(self, theta):
        k1, k2, k3, k4 = self.k
        t2 = theta * theta
        return self.f * theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def _poly_dr(self, theta):
        k1, k2, k3, k4 = self.k
        t2 = theta * theta
        return self.f * (1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4))))

    def _poly_ddr(self, theta):
        k1, k2, k3, k4 = self.k
        t2 = theta * theta
        return self.f * theta * (6 * k1 + t2 * (20 * k2 + t2 * (42 * k3 + t2 * 72 * k4)))

    def _continuation(self):
        """(theta_max, r, r', r'') for the quadratic continuation, with the
        curvature limited so the lens never folds back before 120 deg."""
        tm = math.radians(self.theta_max_deg)
        r0 = float(self._poly_r(tm))
        d0 = max(float(self._poly_dr(tm)), 0.05 * self.f)
        dd0 = float(self._poly_ddr(tm))
        span = max(math.radians(120.0) - tm, 1e-3)
        dd0 = max(dd0, -0.75 * d0 / span)
        return tm, r0, d0, dd0

    def _r_of_theta(self, theta: np.ndarray) -> np.ndarray:
        r = self._poly_r(theta)
        if self.theta_max_deg is None:
            return r
        tm, r0, d0, dd0 = self._continuation()
        dt = theta - tm
        return np.where(theta > tm, r0 + d0 * dt + 0.5 * dd0 * dt * dt, r)

    def _dr_dtheta(self, theta: np.ndarray) -> np.ndarray:
        d = self._poly_dr(theta)
        if self.theta_max_deg is None:
            return d
        tm, r0, d0, dd0 = self._continuation()
        return np.where(theta > tm, d0 + dd0 * (theta - tm), d)

    def _theta_of_r(self, r: np.ndarray) -> np.ndarray:
        theta = r / self.f
        for _ in range(30):
            err = self._r_of_theta(theta) - r
            d = self._dr_dtheta(theta)
            d = np.where(np.abs(d) < 1e-9, 1e-9, d)
            step = err / d
            theta = theta - step
            if np.all(np.abs(step) < 1e-12):
                break
        return theta

    # ----- tangential ------------------------------------------------------
    def _apply_tangential(self, xd, yd):
        p1, p2 = self.p
        if p1 == 0.0 and p2 == 0.0:
            return xd, yd
        r2 = xd * xd + yd * yd
        x = xd + 2 * p1 * xd * yd + p2 * (r2 + 2 * xd * xd)
        y = yd + p1 * (r2 + 2 * yd * yd) + 2 * p2 * xd * yd
        return x, y

    def _remove_tangential(self, x, y):
        p1, p2 = self.p
        if p1 == 0.0 and p2 == 0.0:
            return x, y
        xd, yd = x.copy(), y.copy()
        for _ in range(20):
            xe, ye = self._apply_tangential(xd, yd)
            xd = xd - (xe - x)
            yd = yd - (ye - y)
        return xd, yd

    # ----- public mapping --------------------------------------------------
    def world_to_pixel(self, d_world: np.ndarray):
        """ENU unit vectors (N,3) -> (u, v, theta).  theta in radians."""
        d = np.atleast_2d(d_world) @ self.R.T
        rho = np.hypot(d[:, 0], d[:, 1])
        theta = np.arctan2(rho, d[:, 2])
        phi = np.arctan2(d[:, 1], d[:, 0])
        r = self._r_of_theta(theta) / self.f      # normalised radius
        xd, yd = r * np.cos(phi), r * np.sin(phi)
        xd, yd = self._apply_tangential(xd, yd)
        if self.mirror:
            xd = -xd
        u = self.cx + self.f * xd
        v = self.cy + self.f * self.aspect * yd
        return u, v, theta

    def azel_to_pixel(self, az_deg, el_deg):
        """Returns (u, v, visible) arrays.  ``visible`` is False for directions
        behind the lens (theta > 100 deg) or outside the image."""
        u, v, theta = self.world_to_pixel(azel_to_enu(np.atleast_1d(az_deg), np.atleast_1d(el_deg)))
        visible = (theta < math.radians(100)) & (u >= 0) & (v >= 0) & \
                  (u <= self.image_width - 1) & (v <= self.image_height - 1)
        return u, v, visible

    def pixel_to_world(self, u, v) -> np.ndarray:
        u = np.atleast_1d(np.asarray(u, dtype=float))
        v = np.atleast_1d(np.asarray(v, dtype=float))
        xd = (u - self.cx) / self.f
        yd = (v - self.cy) / (self.f * self.aspect)
        if self.mirror:
            xd = -xd
        xd, yd = self._remove_tangential(xd, yd)
        r = np.hypot(xd, yd) * self.f
        phi = np.arctan2(yd, xd)
        theta = self._theta_of_r(r)
        st = np.sin(theta)
        d_cam = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)
        return d_cam @ self.R          # R^T applied row-wise

    def pixel_to_azel(self, u, v):
        return enu_to_azel(self.pixel_to_world(u, v))

    def extrapolated(self, u, v) -> np.ndarray:
        """True where a pixel lies beyond the off-axis angle covered by the
        calibration stars (the mapping there is extrapolated)."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if self.theta_max_deg is None:
            return np.zeros(u.shape, dtype=bool)
        d_cam = self.pixel_to_world(u, v) @ self.R.T
        theta = np.degrees(np.arccos(np.clip(d_cam[:, 2], -1, 1)))
        return theta > self.theta_max_deg

    # ----- resolution scaling ---------------------------------------------
    def scaled(self, width: int, height: int) -> "FisheyeModel":
        """Same lens at a different (uniformly or not) resized resolution."""
        sx = width / self.image_width
        sy = height / self.image_height
        return FisheyeModel(width, height,
                            (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5,
                            self.f * sx, self.aspect * sy / sx,
                            list(self.k), list(self.p), list(self.rvec), self.mirror,
                            self.theta_max_deg)

    # ----- human readable summary -----------------------------------------
    def describe(self) -> dict:
        R = self.R
        axis_world = R.T @ np.array([0.0, 0.0, 1.0])
        axis_az, axis_el = enu_to_azel(axis_world)
        # direction of true north in the image, measured at the optical axis
        n = np.array([0.0, 1.0, 0.0])
        t = n - np.dot(n, axis_world) * axis_world
        if np.linalg.norm(t) < 1e-6:          # axis points at the north horizon
            t = np.array([0.0, 0.0, 1.0])
        d1 = axis_world + 0.02 * t / np.linalg.norm(t)
        u0, v0, _ = self.world_to_pixel(axis_world)
        u1, v1, _ = self.world_to_pixel(d1 / np.linalg.norm(d1))
        north_img_angle = math.degrees(math.atan2(float(u1[0] - u0[0]), float(-(v1[0] - v0[0])))) % 360
        theta_edge = float(self._theta_of_r(np.array([min(self.image_width, self.image_height) / 2.0]))[0])
        return {
            "optical_axis_azimuth_deg": float(axis_az),
            "optical_axis_elevation_deg": float(axis_el),
            "tilt_from_zenith_deg": 90.0 - float(axis_el),
            "north_direction_in_image_deg_clockwise_from_up": north_img_angle,
            "fov_across_short_side_deg": 2 * math.degrees(theta_edge),
            "mirror": bool(self.mirror),
        }

    # ----- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["k"] = [float(x) for x in self.k]
        d["p"] = [float(x) for x in self.p]
        d["rvec"] = [float(x) for x in self.rvec]
        for key in ("cx", "cy", "f", "aspect"):
            d[key] = float(d[key])
        if self.theta_max_deg is not None:
            d["theta_max_deg"] = float(self.theta_max_deg)
        d["model"] = "kannala_brandt_enu_v1"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FisheyeModel":
        d = dict(d)
        d.pop("model", None)
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "FisheyeModel":
        return cls.from_dict(json.loads(s))
