import io
import json
from datetime import datetime, timedelta, timezone

import pytest

from bob_startracker.sky_conditions import (SkyMonitor, clear_dark_windows, fetch_forecast,
                                            parse_open_meteo)
from bob_startracker.star_catalog import Site

SITE = Site(51.2, 7.1, 150.0)
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)   # local winter noon


def _doc(clouds, start=T0):
    """Open-Meteo style response (timeformat=unixtime)."""
    times = [int((start + timedelta(hours=i)).timestamp()) for i in range(len(clouds))]
    return {"latitude": SITE.latitude_deg, "longitude": SITE.longitude_deg,
            "hourly": {"time": times, "cloud_cover": clouds,
                       "cloud_cover_low": clouds, "cloud_cover_mid": [0] * len(clouds),
                       "cloud_cover_high": [0] * len(clouds),
                       "precipitation_probability": [0] * len(clouds),
                       "visibility": [24000] * len(clouds)},
            "hourly_units": {"cloud_cover": "%"}}


# overcast until 22 UTC, clear 22-03 UTC, cloudy again afterwards
CLOUDS = [90] * 10 + [5] * 5 + [80] * 33


def test_parse_and_fetch_builds_request():
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def opener(req, timeout):
        seen["url"] = req.full_url
        return Resp(json.dumps(_doc(CLOUDS)).encode())

    fc = fetch_forecast(SITE, opener=opener)
    assert "latitude=51.2000" in seen["url"] and "cloud_cover" in seen["url"]
    assert "timeformat=unixtime" in seen["url"]
    assert len(fc) == len(CLOUDS) and fc[0].time_utc == T0 and fc[10].cloud_cover == 5


def test_clear_dark_window_found():
    fc = parse_open_meteo(_doc(CLOUDS))
    wins = clear_dark_windows(SITE, fc, T0, hours=36)
    assert len(wins) == 1
    w = wins[0]
    # clear slot is 22:00-03:00 UTC (+/- half an hour of slot rounding)
    assert abs((w.start_utc - (T0 + timedelta(hours=10))).total_seconds()) <= 1800
    assert abs((w.end_utc - (T0 + timedelta(hours=15))).total_seconds()) <= 1800
    assert w.max_cloud_cover <= 20


def _monitor(**kw):
    fc = parse_open_meteo(_doc(CLOUDS))
    return SkyMonitor(SITE, fetcher=lambda site, url: fc, **kw)


def test_should_capture_uses_sun_forecast_and_camera():
    m = _monitor()
    ok, why = m.should_capture(T0)                          # daytime
    assert not ok and "sun" in why
    ok, why = m.should_capture(T0 + timedelta(hours=8))     # dark but overcast
    assert not ok and "forecast" in why
    night = T0 + timedelta(hours=12)                        # dark and clear
    assert m.should_capture(night)[0]
    m.update_local_cloud(0.6, now=night.timestamp())        # camera disagrees
    ok, why = m.should_capture(night)
    assert not ok and "camera" in why
    m.update_local_cloud(5.0, now=night.timestamp())        # percent tolerated -> 5 %
    assert m.should_capture(night)[0]


@pytest.mark.parametrize("policy,expected", [("allow", True), ("skip", False)])
def test_offline_policy(policy, expected):
    def boom(site, url):
        raise OSError("network unreachable")
    m = SkyMonitor(SITE, fetcher=boom, offline_policy=policy)
    ok, why = m.should_capture(T0 + timedelta(hours=12))
    assert ok is expected
    assert "network unreachable" in m.conditions(T0)["forecast_error"]
