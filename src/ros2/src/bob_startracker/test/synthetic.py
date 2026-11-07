"""Render a synthetic all-sky star frame with a *different* lens law
(equisolid) from the fitted polynomial model, so the tests check that the
model can actually absorb real lens distortion."""
import math
import numpy as np
from bob_startracker.fisheye_model import rodrigues, rotation_to_rvec, zenith_rotation, azel_to_enu


class TrueLens:
    def __init__(self, w=1440, h=1440, cx=731.3, cy=703.8, fov_deg=190.0, radius=700.0,
                 yaw_deg=137.0, tilt_deg=2.5, tilt_dir_deg=40.0, mirror=False, aspect=1.0):
        self.w, self.h, self.cx, self.cy, self.mirror, self.aspect = w, h, cx, cy, mirror, aspect
        # equisolid: r = 2 f sin(theta/2)
        self.f = radius / (2 * math.sin(math.radians(fov_deg / 2) / 2))
        R = zenith_rotation(yaw_deg)
        t = math.radians(tilt_deg); a = math.radians(tilt_dir_deg)
        tilt = rodrigues(np.array([math.cos(a), math.sin(a), 0.0]) * t)
        self.R = tilt @ R

    def project(self, az, el):
        d = azel_to_enu(np.asarray(az), np.asarray(el)) @ self.R.T
        theta = np.arctan2(np.hypot(d[:, 0], d[:, 1]), d[:, 2])
        phi = np.arctan2(d[:, 1], d[:, 0])
        r = 2 * self.f * np.sin(theta / 2)
        x = r * np.cos(phi)
        if self.mirror:
            x = -x
        return self.cx + x, self.cy + self.aspect * r * np.sin(phi)


def _blob(img, uu, vv, amp, psf_sigma):
    x0, y0 = int(uu), int(vv)
    ys, xs = np.mgrid[y0 - 5:y0 + 6, x0 - 5:x0 + 6]
    img[y0 - 5:y0 + 6, x0 - 5:x0 + 6] += amp * np.exp(-((xs - uu) ** 2 + (ys - vv) ** 2) / (2 * psf_sigma ** 2))


def render(lens, stars, rng, noise=3.0, sky=20.0, n_false=40, psf_sigma=1.1, n_field=0, n_hot=10):
    """n_field: uncatalogued faint field stars (mag 3.5-5.5) as distractors.
    n_false / n_hot: small bright blobs (aircraft lights, hot pixels)."""
    img = np.full((lens.h, lens.w), sky, np.float32)
    # vignetting + horizon glow
    yy, xx = np.mgrid[0:lens.h, 0:lens.w]
    rr = np.hypot(xx - lens.cx, yy - lens.cy)
    img += 15.0 * (rr / 700.0) ** 4
    img[rr > 705] = 2.0
    u, v = lens.project([s.az_deg for s in stars], [s.el_deg for s in stars])
    truth = []
    for s, uu, vv in zip(stars, u, v):
        if s.el_deg < 0 or not (5 < uu < lens.w - 5 and 5 < vv < lens.h - 5):
            continue
        amp = 220.0 * 10 ** (-0.4 * (s.mag - 0.0)) * (0.5 + 0.5 * math.sin(math.radians(max(s.el_deg, 1))))
        amp = min(amp, 400.0)
        _blob(img, uu, vv, amp, psf_sigma)
        truth.append((s.name, uu, vv))
    for _ in range(n_field):
        rr_ = 690 * math.sqrt(rng.uniform()); a_ = rng.uniform(0, 2 * math.pi)
        uu, vv = lens.cx + rr_ * math.cos(a_), lens.cy + rr_ * math.sin(a_)
        if 6 < uu < lens.w - 6 and 6 < vv < lens.h - 6:
            mag = 5.5 - 2.0 * rng.uniform() ** 2
            _blob(img, uu, vv, 220.0 * 10 ** (-0.4 * mag), psf_sigma)
    for _ in range(n_false + n_hot):   # hot pixels / aircraft lights
        x0, y0 = rng.integers(20, lens.w - 20), rng.integers(20, lens.h - 20)
        if np.hypot(x0 - lens.cx, y0 - lens.cy) < 690:
            img[y0 - 1:y0 + 2, x0 - 1:x0 + 2] += rng.uniform(30, 120)
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8), truth
