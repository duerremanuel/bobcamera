"""Star (and bright planet) catalog -> apparent topocentric az/el.

The default catalog is the ~115 bright named stars that ship with PyEphem
(Hipparcos positions + proper motion, down to about mag 3), optionally
supplemented with the naked-eye planets.  PyEphem handles precession,
nutation, aberration and atmospheric refraction, and its azimuth is measured
from TRUE north.

A larger catalog can be supplied as CSV with the columns
    name,ra_deg,dec_deg,mag[,pmra_mas_yr,pmdec_mas_yr]
(J2000 / ICRS coordinates), for example an export of the Yale Bright Star
Catalogue or Hipparcos filtered to mag < 5.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import ephem
import ephem.stars


@dataclass
class Site:
    latitude_deg: float
    longitude_deg: float          # east positive
    elevation_m: float = 0.0
    pressure_hpa: float = 1010.0  # 0 disables refraction
    temperature_c: float = 10.0

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class CatalogStar:
    name: str
    mag: float
    az_deg: float
    el_deg: float
    kind: str = "star"            # star | planet


PLANETS = {
    "Venus": ephem.Venus, "Jupiter": ephem.Jupiter, "Mars": ephem.Mars,
    "Saturn": ephem.Saturn, "Mercury": ephem.Mercury,
}


def _observer(site: Site, when: datetime) -> ephem.Observer:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    obs = ephem.Observer()
    obs.lat = str(site.latitude_deg)
    obs.lon = str(site.longitude_deg)
    obs.elevation = site.elevation_m
    obs.pressure = site.pressure_hpa
    obs.temp = site.temperature_c
    obs.date = ephem.Date(when.astimezone(timezone.utc).replace(tzinfo=None))
    obs.epoch = obs.date
    return obs


class StarCatalog:

    def __init__(self, csv_path: Optional[str] = None, include_planets: bool = True,
                 include_builtin: bool = True):
        self._bodies = []   # (name, body, kind)
        if include_builtin:
            for name in sorted(ephem.stars.stars):
                self._bodies.append((name, ephem.star(name), "star"))
        if csv_path:
            known = {n.lower() for n, _, _ in self._bodies}
            for row in self._read_csv(csv_path):
                if row["name"].lower() in known:
                    continue
                self._bodies.append((row["name"], row["body"], "star"))
        if include_planets:
            for name, cls in PLANETS.items():
                self._bodies.append((name, cls(), "planet"))

    @staticmethod
    def _read_csv(path):
        out = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                body = ephem.FixedBody()
                body._ra = math.radians(float(row["ra_deg"]))
                body._dec = math.radians(float(row["dec_deg"]))
                body._epoch = ephem.J2000
                body.mag = float(row.get("mag", 5.0) or 5.0)
                if row.get("pmra_mas_yr"):
                    # ephem expects mas/yr in RA *on the sky* (already cos(dec) scaled)
                    body._pmra = float(row["pmra_mas_yr"])
                    body._pmdec = float(row.get("pmdec_mas_yr") or 0.0)
                out.append({"name": row["name"], "body": body})
        return out

    def __len__(self):
        return len(self._bodies)

    def names(self):
        return [n for n, _, _ in self._bodies]

    def visible(self, site: Site, when: datetime, min_elevation_deg: float = 0.0,
                max_mag: float = 6.0) -> List[CatalogStar]:
        obs = _observer(site, when)
        out = []
        for name, body, kind in self._bodies:
            body.compute(obs)
            el = math.degrees(float(body.alt))
            mag = float(body.mag)
            if el < min_elevation_deg or mag > max_mag:
                continue
            out.append(CatalogStar(name, mag, math.degrees(float(body.az)) % 360.0, el, kind))
        out.sort(key=lambda s: s.mag)
        return out

    def position(self, name: str, site: Site, when: datetime) -> Optional[CatalogStar]:
        obs = _observer(site, when)
        for n, body, kind in self._bodies:
            if n.lower() == name.lower():
                body.compute(obs)
                return CatalogStar(n, float(body.mag), math.degrees(float(body.az)) % 360.0,
                                   math.degrees(float(body.alt)), kind)
        return None

    @staticmethod
    def sun_elevation(site: Site, when: datetime) -> float:
        obs = _observer(site, when)
        sun = ephem.Sun(obs)
        return math.degrees(float(sun.alt))
