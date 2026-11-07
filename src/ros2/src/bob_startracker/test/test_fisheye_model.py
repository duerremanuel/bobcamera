import numpy as np
import pytest

from bob_startracker.fisheye_model import FisheyeModel, angular_separation_deg, rotation_to_rvec, rodrigues


def _model(mirror=False):
    m = FisheyeModel.nominal(1920, 1080, fov_deg=200, up_azimuth_deg=-60, mirror=mirror, radius_px=760)
    m.k = [0.02, -0.004, 0.0005, 0.0]
    m.p = [1e-4, -2e-4]
    m.aspect = 1.003
    m.rvec = list(rotation_to_rvec(rodrigues(np.array([0.03, -0.02, 0.0])) @ m.R))
    return m


@pytest.mark.parametrize("mirror", [False, True])
def test_roundtrip_azel_pixel(mirror):
    m = _model(mirror)
    az, el = np.meshgrid(np.arange(0, 360, 7.0), np.arange(1, 90, 4.0))
    az, el = az.ravel(), el.ravel()
    u, v, vis = m.azel_to_pixel(az, el)
    a2, e2 = m.pixel_to_azel(u[vis], v[vis])
    assert vis.sum() > 100
    assert angular_separation_deg(a2, e2, az[vis], el[vis]).max() < 1e-6


def test_zenith_and_north():
    m = FisheyeModel.nominal(1000, 1000, 180, up_azimuth_deg=0.0)
    az, el = m.pixel_to_azel([m.cx], [m.cy])
    assert el[0] == pytest.approx(90.0, abs=1e-9)
    # north at the top of the image, east on the LEFT for an upward-looking camera
    u, v, _ = m.azel_to_pixel([0.0, 90.0], [0.0, 0.0])
    assert v[0] < m.cy and abs(u[0] - m.cx) < 1e-6
    assert u[1] < m.cx and abs(v[1] - m.cy) < 1e-6
    assert m.describe()["north_direction_in_image_deg_clockwise_from_up"] == pytest.approx(0.0, abs=1e-6)


def test_scaled_model_matches_resized_image():
    m = _model()
    s = m.scaled(960, 540)
    u, v, _ = m.azel_to_pixel([123.0], [37.0])
    u2, v2, _ = s.azel_to_pixel([123.0], [37.0])
    assert u2[0] == pytest.approx((u[0] + 0.5) * 0.5 - 0.5, abs=1e-6)
    assert v2[0] == pytest.approx((v[0] + 0.5) * 0.5 - 0.5, abs=1e-6)


def test_json_roundtrip():
    m = _model(True)
    m2 = FisheyeModel.from_json(m.to_json())
    u, v, _ = m.azel_to_pixel([10.0, 200.0], [20.0, 70.0])
    u2, v2, _ = m2.azel_to_pixel([10.0, 200.0], [20.0, 70.0])
    np.testing.assert_allclose([u, v], [u2, v2])
