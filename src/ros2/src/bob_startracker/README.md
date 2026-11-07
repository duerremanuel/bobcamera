# bob_startracker

Star-tracker calibration for Bob's all-sky fisheye camera. It finds the stars in
captured night frames, stores where they are on the image, identifies them from
the camera site and the capture time, and fits a distortion-corrected model.
That model converts **image pixels ⇄ true-north azimuth / elevation**.

```
night frame ──► stack N frames ──► detect stars (sub-pixel) ──► store capture
                                                                     │
catalog (PyEphem bright stars + planets, refraction, true north) ◄───┤
                                                                     ▼
  brute-force yaw/mirror/scale ─► iterative match + least squares ─► calibration.json
                                                                     │
          PixelToAzEl / AzElToPixel services, tracks_azel topic ◄────┘
```

## Conventions

* **Azimuth** is measured from **true (geographic) north**, clockwise through east,
  in degrees `[0, 360)`. It is not magnetic north.
* **Elevation** is the *apparent* elevation above the horizon. Catalog positions
  include atmospheric refraction, which is what the camera actually sees.
* Pixels use OpenCV conventions: `(0,0)` is the centre of the top-left pixel, x
  runs right and y runs down.
* Time is always **UTC**.

## Lens model

The lens is modelled as a Kannala–Brandt (OpenCV-fisheye style) radial polynomial
plus a full 3-D camera rotation:

```
theta = angle of the ray from the optical axis
r     = f · (θ + k1θ³ + k2θ⁵ + k3θ⁷ + k4θ⁹)
u     = cx ± r·cos φ          (− if the image is mirrored)
v     = cy + aspect · r·sin φ
```

The model also allows optional tangential terms (p1, p2). It covers
equidistant, equisolid and stereographic fisheyes up to about 200° FOV. It
estimates camera tilt, the principal-point offset and whether the image is
mirrored.

The fit only frees parameters that the data can support. Higher radial terms
need both enough stars and stars that reach far enough off-axis. The lens is
constrained to be monotonic, so it cannot fold back on itself. Beyond the
outermost calibration star, the curve continues quadratically instead of
trusting the polynomial. The calibration records `min_star_elevation_deg`, and
`PixelToAzEl` returns `extrapolated: true` for pixels outside the region
covered by stars.

## Workflow (ROS)

1. Set the camera site in the launch config, under `basic_config: startracker_node`.
   Use `latitude` and `longitude` (WGS84, east positive), plus `elevation_m`.
2. On a clear night (sun below about −12°), request a capture. Add `calibrate: true`
   to fit the model at the same time:

   ```bash
   ros2 service call /bob/startracker/capture bob_interfaces/srv/StarTrackerCapture "{calibrate: true}"
   ```

   A single capture is usually enough. A second capture a few hours later puts
   stars on different pixels and tightens the distortion fit. You can also set
   `auto_capture_interval_s` (for example 3600) to collect captures unattended
   through the night.
   Auto-captures only run when the sky is **probably clear**. Two signals are
   used (see below).
3. Use the calibration:

   ```bash
   ros2 service call /bob/startracker/pixel_to_azel bob_interfaces/srv/PixelToAzEl "{x: [960.0], y: [540.0]}"
   ros2 service call /bob/startracker/azel_to_pixel bob_interfaces/srv/AzElToPixel "{azimuth_deg: [180.0], elevation_deg: [45.0]}"
   ros2 topic echo /bob/startracker/tracks_azel          # every tracked object in az/el
   ros2 topic echo /bob/startracker/calibration          # latched calibration JSON
   ```

   Both services take `image_width` and `image_height` when your pixels come from
   a resized frame.

**Recorded videos** contain no real capture time. Set `time_override` to the UTC
time the video was recorded. If the camera clock is off, correct it with
`time_offset_s`: a 1-minute error moves the stars by about 0.25°.

## When is the sky clear? (weather)

Two independent signals decide whether an automatic capture runs:

| signal | source | answers | setting |
|---|---|---|---|
| cloud forecast | [Open-Meteo](https://open-meteo.com) hourly `cloud_cover` (free for non-commercial use, no API key) | *when* tonight is likely clear | `weather_max_cloud_cover_pct` (20) |
| camera cloud estimate | Bob's `cloud_estimator` (`bob/observer/cloud_estimation`) | is it clear *right now* above this camera | `camera_max_cloud_fraction` (0.2) |

Both must pass, and the sun must be below `auto_capture_max_sun_elevation_deg`
(−12°). The forecast is refreshed every 30 min. Each capture stores the
conditions at capture time in `capture.json`.

If the internet is unreachable, `weather_offline_policy` decides what happens.
`allow` captures on darkness and the camera estimate alone; `skip` waits until a
forecast is available. Manual captures always run, and they warn if the
forecast says it is cloudy.

The node publishes `bob/startracker/sky_outlook` (latched JSON). It contains
the current forecast and camera estimate, plus the **clear dark windows** for
the next 36 hours. Offline, you can check the same thing with:

```bash
bob_startracker outlook --lat 51.2 --lon 7.1
```

## Workflow (offline CLI, no ROS)

```bash
bob_startracker capture night.png --time 2026-01-15T21:00:00Z --lat 51.2 --lon 7.1 \
    --elev 150 --mask assets/masks/detection-mask.jpg
bob_startracker list
bob_startracker add-star 20260115T210000Z Vega 812.4 233.9   # optional manual pin
bob_startracker calibrate
bob_startracker px2azel 960 540
bob_startracker azel2px 180 45
bob_startracker grid check.jpg      # az/el grid + catalog stars over the last capture
```

Inside the dev container, `python3 -m bob_startracker.cli …` does the same.

## Files (`storage_folder`, default `assets/startracker`)

| file | content |
|---|---|
| `captures/<id>/image.png` | stacked frame used for detection |
| `captures/<id>/capture.json` | UTC time, site, every detection (x, y, flux, SNR, FWHM), identified stars with catalog az/el and residuals, manual pins |
| `captures/<id>/overlay.jpg` | detections (red), identified stars (green), manual pins (yellow), predicted catalog stars (blue +), az/el grid with N/E/S/W |
| `calibration.json` | the model, a human-readable summary (tilt, north direction in the image, FOV), fit statistics and quality |
| `calibration_history/` | previous calibrations |

`calibration.json` is plain JSON. Other tools, such as the digital twin or a PTZ
slew controller, can re-implement the model in a few lines; see `fisheye_model.py`.

## Accuracy and limits

* On synthetic all-sky frames rendered through a *different* lens law
  (equisolid), the recovered mapping is accurate to about 0.01° RMS over the
  whole sky. The tests in `test/` check this, along with mirrored images, cropped
  16:9 sensors, dense star fields and obstructed horizons.
* The built-in catalog is PyEphem's ~115 bright named stars (to about mag 3) plus
  Mercury to Saturn. To add fainter stars, supply `catalog_csv` with the columns
  `name,ra_deg,dec_deg,mag[,pmra_mas_yr,pmdec_mas_yr]` (J2000), for example from
  the Yale Bright Star Catalogue.
* Use Bob's detection mask (`mask_file`) to hide trees, buildings and lights.
  The black area outside the fisheye circle is excluded automatically.
* A solution rated `poor` is never written to `calibration.json` unless you pass
  `force`. This happens with a wrong time, wrong site or cloud cover.

## Tests

```bash
cd src/ros2/src/bob_startracker && python3 -m pytest -q test
```
