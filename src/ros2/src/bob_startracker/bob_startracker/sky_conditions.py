"""Is the sky (probably) clear and dark enough for a star capture?

Two independent signals:

* a cloud-cover **forecast** from Open-Meteo (https://open-meteo.com, free for
  non-commercial use, no API key): tells us *when* tonight is likely clear,
  so auto-captures only happen in clear dark windows;
* Bob's own **camera-based cloud estimate** (bob_observer cloud_estimator,
  topic ``bob/observer/cloud_estimation``): tells us whether it is clear
  *right now* over this camera, which a forecast can't guarantee.

Network failures never raise into the caller: the forecast is simply
``None`` and the caller decides (``offline_policy``) what to do.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional

from .star_catalog import Site, StarCatalog

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = ("cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
               "precipitation_probability", "visibility")


@dataclass
class HourForecast:
    time_utc: datetime
    cloud_cover: float                       # % total
    cloud_cover_low: Optional[float] = None
    cloud_cover_mid: Optional[float] = None
    cloud_cover_high: Optional[float] = None
    precipitation_probability: Optional[float] = None
    visibility_m: Optional[float] = None

    def to_dict(self):
        d = asdict(self)
        d["time_utc"] = self.time_utc.isoformat()
        return d


@dataclass
class ClearWindow:
    start_utc: datetime
    end_utc: datetime
    mean_cloud_cover: float
    max_cloud_cover: float

    @property
    def hours(self) -> float:
        return (self.end_utc - self.start_utc).total_seconds() / 3600.0

    def to_dict(self):
        return {"start_utc": self.start_utc.isoformat(), "end_utc": self.end_utc.isoformat(),
                "hours": round(self.hours, 1), "mean_cloud_cover": round(self.mean_cloud_cover, 1),
                "max_cloud_cover": round(self.max_cloud_cover, 1)}


def parse_open_meteo(doc: dict) -> List[HourForecast]:
    """Parse an Open-Meteo /v1/forecast response requested with
    ``timeformat=unixtime`` (times in UTC seconds)."""
    h = doc["hourly"]
    times = h["time"]
    out = []
    for i, t in enumerate(times):
        if isinstance(t, (int, float)):
            tt = datetime.fromtimestamp(t, tz=timezone.utc)
        else:   # ISO without zone, only valid when timezone=GMT (the default)
            tt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        get = lambda k: (h.get(k) or [None] * len(times))[i]
        cc = get("cloud_cover")
        if cc is None:
            continue
        out.append(HourForecast(tt, float(cc), get("cloud_cover_low"), get("cloud_cover_mid"),
                                get("cloud_cover_high"), get("precipitation_probability"),
                                get("visibility")))
    return out


def fetch_forecast(site: Site, hours: int = 48, url: str = OPEN_METEO_URL,
                   timeout_s: float = 10.0,
                   opener: Callable = urllib.request.urlopen) -> List[HourForecast]:
    """Hourly cloud forecast for the site.  Raises on network/parse errors."""
    days = max(1, min(16, (hours + 23) // 24 + 1))
    q = urllib.parse.urlencode({
        "latitude": f"{site.latitude_deg:.4f}",
        "longitude": f"{site.longitude_deg:.4f}",
        "hourly": ",".join(HOURLY_VARS),
        "timeformat": "unixtime",
        "forecast_days": days,
        "past_days": 1,
    })
    req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": "bobcamera-startracker"})
    with opener(req, timeout=timeout_s) as resp:
        doc = json.loads(resp.read().decode("utf-8"))
    if "hourly" not in doc:
        raise RuntimeError(f"unexpected weather response: {doc.get('reason', doc)}")
    return parse_open_meteo(doc)


def forecast_at(forecast: List[HourForecast], when: datetime) -> Optional[HourForecast]:
    """Hour slot containing ``when`` (Open-Meteo values describe the hour at
    their timestamp; we take the nearest slot within 90 min)."""
    if not forecast:
        return None
    best = min(forecast, key=lambda f: abs((f.time_utc - when).total_seconds()))
    return best if abs((best.time_utc - when).total_seconds()) <= 5400 else None


def clear_dark_windows(site: Site, forecast: List[HourForecast], start: datetime,
                       hours: int = 36, max_cloud_cover: float = 20.0,
                       max_sun_elevation_deg: float = -12.0,
                       max_precip_probability: float = 30.0,
                       min_hours: float = 1.0) -> List[ClearWindow]:
    """Contiguous periods where it is dark (sun below the limit) and the
    forecast is clear.  Evaluated in 15-minute steps."""
    windows: List[ClearWindow] = []
    cur, clouds = None, []
    step = timedelta(minutes=15)
    t = start.astimezone(timezone.utc)
    end = t + timedelta(hours=hours)
    while t <= end:
        f = forecast_at(forecast, t)
        ok = (f is not None and f.cloud_cover <= max_cloud_cover
              and (f.precipitation_probability is None or f.precipitation_probability <= max_precip_probability)
              and StarCatalog.sun_elevation(site, t) <= max_sun_elevation_deg)
        if ok:
            if cur is None:
                cur, clouds = t, []
            clouds.append(f.cloud_cover)
        elif cur is not None:
            windows.append(ClearWindow(cur, t, sum(clouds) / len(clouds), max(clouds)))
            cur = None
        t += step
    if cur is not None:
        windows.append(ClearWindow(cur, t, sum(clouds) / len(clouds), max(clouds)))
    return [w for w in windows if w.hours >= min_hours]


class SkyMonitor:
    """Caches the forecast and the latest camera cloud estimate and decides
    whether an automatic capture should run now."""

    def __init__(self, site: Site, enabled: bool = True, url: str = OPEN_METEO_URL,
                 refresh_s: float = 1800.0, max_cloud_cover: float = 20.0,
                 max_sun_elevation_deg: float = -12.0, local_max_cloud_fraction: float = 0.2,
                 local_max_age_s: float = 600.0, offline_policy: str = "allow",
                 fetcher: Callable = fetch_forecast):
        self.site = site
        self.enabled = enabled
        self.url = url
        self.refresh_s = refresh_s
        self.max_cloud_cover = max_cloud_cover
        self.max_sun_elevation_deg = max_sun_elevation_deg
        self.local_max_cloud_fraction = local_max_cloud_fraction
        self.local_max_age_s = local_max_age_s
        self.offline_policy = offline_policy          # allow | skip
        self.fetcher = fetcher
        self.forecast: Optional[List[HourForecast]] = None
        self.fetched_at = 0.0
        self.last_error = ""
        self.local_cloud: Optional[float] = None      # fraction 0..1
        self.local_cloud_at = 0.0

    # ----- inputs ---------------------------------------------------------
    def update_local_cloud(self, value: float, now: Optional[float] = None):
        v = float(value)
        if v > 1.0:            # tolerate a percentage
            v /= 100.0
        self.local_cloud = v
        self.local_cloud_at = now if now is not None else time.time()

    def refresh(self, force: bool = False, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        if not self.enabled:
            return False
        if not force and self.forecast is not None and now - self.fetched_at < self.refresh_s:
            return True
        try:
            self.forecast = self.fetcher(self.site, url=self.url)
            self.fetched_at = now
            self.last_error = ""
            return True
        except Exception as e:   # offline, DNS, API error ...
            self.last_error = str(e)
            # keep a stale forecast for up to 12 h rather than nothing
            if self.forecast is not None and now - self.fetched_at > 12 * 3600:
                self.forecast = None
            return False

    # ----- decisions --------------------------------------------------------
    def forecast_now(self, when: datetime) -> Optional[HourForecast]:
        return forecast_at(self.forecast or [], when)

    def local_cloud_now(self, now: Optional[float] = None) -> Optional[float]:
        now = now if now is not None else time.time()
        if self.local_cloud is None or now - self.local_cloud_at > self.local_max_age_s:
            return None
        return self.local_cloud

    def should_capture(self, when: Optional[datetime] = None) -> tuple:
        """Returns (ok: bool, reason: str)."""
        when = when or datetime.now(timezone.utc)
        sun = StarCatalog.sun_elevation(self.site, when)
        if sun > self.max_sun_elevation_deg:
            return False, f"sun at {sun:.1f} deg (limit {self.max_sun_elevation_deg})"
        local = self.local_cloud_now(when.timestamp())
        if local is not None and local > self.local_max_cloud_fraction:
            return False, f"camera sees {local * 100:.0f}% cloud"
        if self.enabled:
            self.refresh(now=when.timestamp())
            f = self.forecast_now(when)
            if f is None:
                if self.offline_policy == "skip":
                    return False, f"no weather forecast ({self.last_error or 'unavailable'})"
            elif f.cloud_cover > self.max_cloud_cover:
                return False, f"forecast {f.cloud_cover:.0f}% cloud (limit {self.max_cloud_cover:.0f}%)"
        return True, "clear and dark" + ("" if local is None else f", camera {local * 100:.0f}% cloud")

    def conditions(self, when: Optional[datetime] = None) -> dict:
        """Snapshot stored with each capture and published as the outlook."""
        when = when or datetime.now(timezone.utc)
        f = self.forecast_now(when) if self.forecast else None
        d = {
            "time_utc": when.isoformat(),
            "sun_elevation_deg": round(StarCatalog.sun_elevation(self.site, when), 2),
            "forecast": f.to_dict() if f else None,
            "camera_cloud_fraction": self.local_cloud_now(when.timestamp()),
            "forecast_source": "Open-Meteo" if self.enabled else None,
            "forecast_error": self.last_error or None,
        }
        if self.forecast:
            d["clear_dark_windows"] = [w.to_dict() for w in clear_dark_windows(
                self.site, self.forecast, when, 36, self.max_cloud_cover, self.max_sun_elevation_deg)]
        return d
