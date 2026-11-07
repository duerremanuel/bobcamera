"""End-to-end: render synthetic all-sky frames through an equisolid lens (not
the model's own polynomial), then detect, identify and fit, and compare the
recovered pixel -> az/el mapping against the true lens over the whole sky."""
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synthetic import TrueLens, render  # noqa: E402
from bob_startracker import CalibrationStore, Site, StarCatalog, angular_separation_deg  # noqa: E402

SITE = Site(51.2, 7.1, 150.0)
CAT = StarCatalog()


def _sky_error(lens, model, min_el=5.0):
    az, el = np.meshgrid(np.arange(0, 360, 5.0), np.arange(min_el, 90, 3.0))
    az, el = az.ravel(), el.ravel()
    u, v = lens.project(az, el)
    ok = (u > 0) & (v > 0) & (u < lens.w) & (v < lens.h)
    a2, e2 = model.pixel_to_azel(u[ok], v[ok])
    return angular_separation_deg(a2, e2, az[ok], el[ok])


def _calibrate(tmp_path, lens, times, seed=0, **render_kw):
    rng = np.random.default_rng(seed)
    store = CalibrationStore(str(tmp_path), CAT)
    for t in times:
        img, _ = render(lens, CAT.visible(SITE, t, -1, 6), rng, **render_kw)
        store.new_capture(img, t, SITE)
    return store, store.calibrate()


T1 = datetime(2026, 1, 15, 21, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 16, 3, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize("mirror", [False, True])
def test_full_circle_two_captures(tmp_path, mirror):
    lens = TrueLens(mirror=mirror)
    store, res = _calibrate(tmp_path, lens, [T1, T2])
    err = _sky_error(lens, res.model)
    assert res.quality == "good"
    assert res.model.mirror == mirror
    assert np.sqrt(np.mean(err ** 2)) < 0.03
    assert err.max() < 0.1
    assert res.model.describe()["tilt_from_zenith_deg"] == pytest.approx(2.5, abs=0.1)
    assert os.path.exists(store.calibration_path)
    cap = store.load_capture(store.list_captures()[0])
    assert len(cap.matches) > 20 and all(np.isfinite(m.residual_deg) for m in cap.matches)


def test_cropped_16x9_sensor_dense_field(tmp_path):
    lens = TrueLens(w=1920, h=1080, cx=965, cy=538, radius=760, fov_deg=200, yaw_deg=-75, tilt_deg=4)
    _, res = _calibrate(tmp_path, lens, [datetime(2026, 6, 20, 22, 30, tzinfo=timezone.utc)],
                        n_false=0, n_hot=20, n_field=1500, noise=4, psf_sigma=1.5)
    err = _sky_error(lens, res.model, min_el=10)
    assert res.quality in ("good", "marginal")
    assert np.sqrt(np.mean(err ** 2)) < 0.05


def test_wrong_time_is_not_saved(tmp_path):
    lens = TrueLens()
    rng = np.random.default_rng(0)
    store = CalibrationStore(str(tmp_path), CAT)
    img, _ = render(lens, CAT.visible(SITE, T1, -1, 6), rng)
    store.new_capture(img, T1 + timedelta(hours=5, minutes=17), SITE)   # bad clock
    with pytest.raises(RuntimeError):
        store.calibrate()
    assert not os.path.exists(store.calibration_path)


def test_manual_star_is_kept(tmp_path):
    lens = TrueLens()
    store, _ = _calibrate(tmp_path, lens, [T1])
    cid = store.list_captures()[0]
    s = CAT.position("Capella", SITE, T1)
    u, v = lens.project([s.az_deg], [s.el_deg])
    store.add_manual_star(cid, "capella", float(u[0]), float(v[0]))
    res = store.calibrate()
    cap = store.load_capture(cid)
    assert cap.manual and cap.manual[0].name == "Capella"
    assert cap.manual[0].residual_px < 1.0
    assert res.quality == "good"


@pytest.mark.parametrize("horizon,max_rms_low", [(20, 0.2), (35, 6.0)])
def test_obstructed_horizon_extrapolates_sanely(tmp_path, horizon, max_rms_low):
    """Trees / buildings hide everything below ``horizon``: the fit must stay
    accurate where stars are, and degrade gracefully (never fold back) below."""
    import cv2
    lens = TrueLens(fov_deg=180, radius=690)
    rng = np.random.default_rng(0)
    store = CalibrationStore(str(tmp_path), CAT)
    az = np.linspace(0, 360, 720)
    u, v = lens.project(az, np.full_like(az, float(horizon)))
    for t in (T1, T2):
        img, _ = render(lens, CAT.visible(SITE, t, -1, 6), rng)
        mask = np.zeros(img.shape, np.uint8)
        cv2.fillPoly(mask, [np.stack([u, v], 1).astype(np.int32)], 255)
        store.new_capture(img, t, SITE, mask)
    res = store.calibrate()
    assert res.min_star_elevation_deg > horizon - 3
    assert np.sqrt(np.mean(_sky_error(lens, res.model, min_el=horizon + 3) ** 2)) < 0.05
    low = _sky_error(lens, res.model, min_el=1)
    assert np.sqrt(np.mean(low ** 2)) < max_rms_low
    # horizon pixels are flagged as extrapolated, zenith is not
    hu, hv = lens.project([0.0, 0.0], [2.0, 89.0])
    assert list(res.model.extrapolated(hu, hv)) == [True, False]
