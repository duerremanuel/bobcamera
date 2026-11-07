"""Offline command line tool (no ROS needed).

Examples::

    # store a night frame (time is UTC!) and detect its stars
    bob_startracker capture frame.png --time 2026-01-15T21:00:00Z \\
        --lat 51.2 --lon 7.1 --elev 150 --mask assets/masks/detection-mask.jpg

    # optional: pin a star by hand if the automatic match struggles
    bob_startracker add-star 20260115T210000Z Vega 812.4 233.9

    # fit the fisheye -> az/el model from every stored capture
    bob_startracker calibrate

    # convert
    bob_startracker px2azel 960 540
    bob_startracker azel2px 180 45
    bob_startracker grid out.jpg            # overlay az/el grid on last capture
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2

from .calibration_store import CalibrationStore, parse_time, utcnow
from .star_catalog import Site, StarCatalog
from .star_detector import DetectorParams, detect_stars, find_image_circle
from .star_matcher import SolverParams


def _store(a) -> CalibrationStore:
    return CalibrationStore(a.root, StarCatalog(a.catalog_csv, include_planets=not a.no_planets))


def _solver(a) -> SolverParams:
    sp = SolverParams(fov_deg=a.fov, max_mag=a.max_mag, min_el_deg=a.min_el,
                      fit_tangential=a.tangential, fit_aspect=a.aspect)
    if a.radius:
        sp.radius_px = a.radius
    if a.cx is not None:
        sp.cx, sp.cy = a.cx, a.cy
    return sp


def cmd_capture(a):
    img = cv2.imread(a.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    mask = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE) if a.mask else None
    t = parse_time(a.time) if a.time else utcnow()
    site = Site(a.lat, a.lon, a.elev, a.pressure, a.temp)
    st = _store(a)
    cap = st.new_capture(img, t, site, mask, DetectorParams(snr=a.snr), notes=a.notes or "")
    st.write_overlay(cap)
    print(json.dumps({"capture_id": cap.capture_id, "detections": len(cap.detections),
                      "sun_elevation_deg": round(cap.sun_elevation_deg, 1)}, indent=2))
    if cap.sun_elevation_deg > -6:
        print("warning: sun is above -6 deg; stars will be hard to see", file=sys.stderr)


def cmd_detect(a):
    img = cv2.imread(a.image, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE) if a.mask else None
    dets = detect_stars(img, mask, DetectorParams(snr=a.snr))
    circle = find_image_circle(img)
    print(json.dumps({"image_circle": circle, "n": len(dets),
                      "detections": [d.to_dict() for d in dets[: a.top]]}, indent=1))


def cmd_add_star(a):
    m = _store(a).add_manual_star(a.capture_id, a.name, a.x, a.y)
    print(json.dumps(m.to_dict(), indent=2))


def cmd_list(a):
    st = _store(a)
    for cid in st.list_captures():
        c = st.load_capture(cid)
        print(f"{cid}  {c.utc_time.isoformat()}  dets={len(c.detections):4d} "
              f"matched={len(c.matches):3d} manual={len(c.manual)} sun={c.sun_elevation_deg:5.1f}")


def cmd_calibrate(a):
    st = _store(a)
    res = st.calibrate(a.captures or None, _solver(a), use_existing=a.use_existing, force=a.force)
    out = {k: v for k, v in res.to_dict().items() if k != "model"}
    out["summary"] = res.model.describe()
    print(json.dumps(out, indent=2))
    print(f"wrote {st.calibration_path}", file=sys.stderr)


def cmd_px2azel(a):
    m = _store(a).load_model()
    if a.width:
        m = m.scaled(a.width, a.height)
    az, el = m.pixel_to_azel([a.x], [a.y])
    print(json.dumps({"azimuth_deg": float(az[0]), "elevation_deg": float(el[0]),
                      "extrapolated": bool(m.extrapolated([a.x], [a.y])[0])}))


def cmd_azel2px(a):
    m = _store(a).load_model()
    if a.width:
        m = m.scaled(a.width, a.height)
    u, v, vis = m.azel_to_pixel([a.az], [a.el])
    print(json.dumps({"x": float(u[0]), "y": float(v[0]), "in_image": bool(vis[0])}))


def cmd_grid(a):
    st = _store(a)
    ids = st.list_captures()
    cid = a.capture_id or (ids[-1] if ids else None)
    if not cid:
        sys.exit("no captures")
    path = st.write_overlay(st.load_capture(cid), st.load_model(), a.output)
    print(path)


def main(argv=None):
    p = argparse.ArgumentParser(prog="bob_startracker", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="assets/startracker", help="storage folder")
    p.add_argument("--catalog-csv", default=None, help="extra stars: name,ra_deg,dec_deg,mag")
    p.add_argument("--no-planets", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="store an image and detect stars")
    c.add_argument("image")
    c.add_argument("--time", help="UTC capture time, ISO 8601 (default: now)")
    c.add_argument("--lat", type=float, required=True)
    c.add_argument("--lon", type=float, required=True, help="east positive")
    c.add_argument("--elev", type=float, default=0.0)
    c.add_argument("--pressure", type=float, default=1010.0, help="hPa, 0 = no refraction")
    c.add_argument("--temp", type=float, default=10.0)
    c.add_argument("--mask", help="bob detection mask (white = sky)")
    c.add_argument("--snr", type=float, default=5.0)
    c.add_argument("--notes")
    c.set_defaults(fn=cmd_capture)

    d = sub.add_parser("detect", help="just detect stars, print JSON")
    d.add_argument("image")
    d.add_argument("--mask")
    d.add_argument("--snr", type=float, default=5.0)
    d.add_argument("--top", type=int, default=50)
    d.set_defaults(fn=cmd_detect)

    m = sub.add_parser("add-star", help="pin a named star to a pixel in a capture")
    m.add_argument("capture_id"); m.add_argument("name")
    m.add_argument("x", type=float); m.add_argument("y", type=float)
    m.set_defaults(fn=cmd_add_star)

    sub.add_parser("list", help="list captures").set_defaults(fn=cmd_list)

    k = sub.add_parser("calibrate", help="fit the model from stored captures")
    k.add_argument("captures", nargs="*")
    k.add_argument("--fov", type=float, default=180.0, help="nominal FOV across --radius")
    k.add_argument("--radius", type=float, help="image circle radius px (default short side/2)")
    k.add_argument("--cx", type=float); k.add_argument("--cy", type=float)
    k.add_argument("--max-mag", type=float, default=4.5)
    k.add_argument("--min-el", type=float, default=5.0)
    k.add_argument("--tangential", action="store_true", help="also fit decentering terms")
    k.add_argument("--aspect", action="store_true", help="also fit pixel aspect ratio")
    k.add_argument("--use-existing", action="store_true", help="start from calibration.json")
    k.add_argument("--force", action="store_true", help="save even a poor solution")
    k.set_defaults(fn=cmd_calibrate)

    for name, fn, args in (("px2azel", cmd_px2azel, ("x", "y")), ("azel2px", cmd_azel2px, ("az", "el"))):
        q = sub.add_parser(name)
        for x in args:
            q.add_argument(x, type=float)
        q.add_argument("--width", type=int, help="resolution of your pixel coords if different")
        q.add_argument("--height", type=int)
        q.set_defaults(fn=fn)

    g = sub.add_parser("grid", help="draw az/el grid + catalog on a capture")
    g.add_argument("output", nargs="?")
    g.add_argument("--capture-id")
    g.set_defaults(fn=cmd_grid)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except RuntimeError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
