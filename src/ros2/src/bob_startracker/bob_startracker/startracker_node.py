"""ROS2 node: star-tracker calibration of the all-sky fisheye camera.

Services
  bob/startracker/capture     bob_interfaces/srv/StarTrackerCapture
  bob/startracker/calibrate   bob_interfaces/srv/StarTrackerCalibrate
  bob/startracker/pixel_to_azel  bob_interfaces/srv/PixelToAzEl
  bob/startracker/azel_to_pixel  bob_interfaces/srv/AzElToPixel

Topics (published)
  bob/startracker/calibration  std_msgs/String (latched JSON of calibration.json)
  bob/startracker/tracks_azel  bob_interfaces/msg/TrackAzElArray

Optionally takes a capture automatically every ``auto_capture_interval_s``
seconds while it is dark (sun below ``auto_capture_max_sun_elevation_deg``)
AND the sky is probably clear: Open-Meteo cloud forecast below
``weather_max_cloud_cover_pct`` and, if Bob's cloud estimator is running, the
camera's own cloud estimate below ``camera_max_cloud_fraction``.

  bob/startracker/sky_outlook  std_msgs/String (JSON: current conditions and
                               the clear dark windows of the next 36 h)
"""

from __future__ import annotations

import os
import json
import threading
import traceback as tb
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

from bob_interfaces.msg import Tracking, TrackAzEl, TrackAzElArray, ObserverCloudEstimation
from bob_interfaces.srv import (StarTrackerCapture, StarTrackerCalibrate,
                                PixelToAzEl, AzElToPixel)

from .calibration_store import CalibrationStore, parse_time
from .fisheye_model import FisheyeModel, angular_separation_deg
from .star_catalog import Site, StarCatalog
from .star_detector import DetectorParams, stack_frames
from .star_matcher import SolverParams
from .sky_conditions import SkyMonitor


class StarTrackerNode(Node):

    def __init__(self):
        super().__init__('bob_startracker')
        self.declare_parameters(namespace='', parameters=[
            ('topic_prefix', 'bob/startracker'),   # per-camera instances: bob/camera1/startracker
            ('image_topic', 'bob/frames/allsky/original'),
            ('tracking_topic', 'bob/tracker/tracking'),
            ('storage_folder', 'assets/startracker'),
            ('latitude', 0.0),
            ('longitude', 0.0),
            ('elevation_m', 0.0),
            ('pressure_hpa', 1010.0),
            ('temperature_c', 10.0),
            ('mask_file', 'assets/masks/detection-mask.jpg'),
            ('stack_frames', 15),
            ('capture_timeout_s', 20.0),
            ('detection_snr', 5.0),
            ('fov_deg', 180.0),
            ('image_circle_radius', 0.0),
            ('max_mag', 4.5),
            ('min_elevation_deg', 5.0),
            ('fit_tangential', False),
            ('catalog_csv', ''),
            ('time_override', ''),       # fixed UTC time for recorded videos
            ('time_offset_s', 0.0),      # added to frame time (camera clock error)
            ('auto_capture_interval_s', 0.0),
            ('auto_capture_max_sun_elevation_deg', -12.0),
            ('weather_enabled', True),                 # Open-Meteo cloud forecast
            ('weather_api_url', 'https://api.open-meteo.com/v1/forecast'),
            ('weather_max_cloud_cover_pct', 20.0),
            ('weather_refresh_s', 1800.0),
            ('weather_offline_policy', 'allow'),       # allow | skip when no forecast
            ('cloud_estimation_topic', 'bob/observer/cloud_estimation'),
            ('camera_max_cloud_fraction', 0.2),
            ('auto_calibrate', True),
            ('tracking_image_width', 0),  # resolution of tracker bboxes; 0 = calibration res.
            ('tracking_image_height', 0),
        ])
        P = lambda n: self.get_parameter(n).value
        self.p = P
        if P('latitude') == 0.0 and P('longitude') == 0.0:
            self.get_logger().warning('latitude/longitude are 0,0 - set the camera site '
                                      'or star identification will fail')

        catalog = StarCatalog(P('catalog_csv') or None)
        self.store = CalibrationStore(P('storage_folder'), catalog)
        self.sky = SkyMonitor(self.site(), enabled=P('weather_enabled'), url=P('weather_api_url'),
                              refresh_s=P('weather_refresh_s'),
                              max_cloud_cover=P('weather_max_cloud_cover_pct'),
                              max_sun_elevation_deg=P('auto_capture_max_sun_elevation_deg'),
                              local_max_cloud_fraction=P('camera_max_cloud_fraction'),
                              offline_policy=P('weather_offline_policy'))
        self.br = CvBridge()
        self.lock = threading.Lock()
        self.model: FisheyeModel | None = None
        self.model_mtime = 0.0
        self.calibration_doc = None

        # frame collection for captures
        self.collect_n = 0
        self.collected = []
        self.collected_stamps = []
        self.collect_done = threading.Event()
        self.busy = threading.Lock()

        sensor_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        tracking_qos = QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE)

        self.cb_frames = MutuallyExclusiveCallbackGroup()
        self.cb_services = ReentrantCallbackGroup()

        self.create_subscription(Image, P('image_topic'), self.image_callback, sensor_qos,
                                 callback_group=self.cb_frames)
        self.create_subscription(ObserverCloudEstimation, P('cloud_estimation_topic'),
                                 lambda m: self.sky.update_local_cloud(m.percentage_cloud_cover),
                                 tracking_qos, callback_group=self.cb_frames)
        self.pub_outlook = self.create_publisher(String, f"{P('topic_prefix')}/sky_outlook", latched)
        self.create_subscription(Tracking, P('tracking_topic'), self.tracking_callback,
                                 tracking_qos, callback_group=self.cb_frames)
        self.pub_calibration = self.create_publisher(String, f"{P('topic_prefix')}/calibration", latched)
        self.pub_tracks = self.create_publisher(TrackAzElArray, f"{P('topic_prefix')}/tracks_azel", 10)

        self.create_service(StarTrackerCapture, f"{P('topic_prefix')}/capture", self.capture_callback,
                            callback_group=self.cb_services)
        self.create_service(StarTrackerCalibrate, f"{P('topic_prefix')}/calibrate", self.calibrate_callback,
                            callback_group=self.cb_services)
        self.create_service(PixelToAzEl, f"{P('topic_prefix')}/pixel_to_azel", self.pixel_to_azel_callback,
                            callback_group=self.cb_services)
        self.create_service(AzElToPixel, f"{P('topic_prefix')}/azel_to_pixel", self.azel_to_pixel_callback,
                            callback_group=self.cb_services)

        self.create_timer(5.0, self.reload_model, callback_group=self.cb_services)
        if P('weather_enabled'):
            self.create_timer(float(P('weather_refresh_s')), self.publish_outlook,
                              callback_group=self.cb_services)
        if P('auto_capture_interval_s') > 0:
            self.create_timer(float(P('auto_capture_interval_s')), self.auto_capture,
                              callback_group=self.cb_services)
        self.reload_model()
        self.publish_outlook()
        self.get_logger().info(f'{self.get_name()} up; storage {self.store.root}, '
                               f'calibrated={self.model is not None}')

    # ------------------------------------------------------------------ helpers
    def site(self) -> Site:
        p = self.p
        return Site(p('latitude'), p('longitude'), p('elevation_m'), p('pressure_hpa'), p('temperature_c'))

    def reload_model(self):
        path = self.store.calibration_path
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        if mtime == self.model_mtime:
            return
        try:
            doc = self.store.load_calibration()
            with self.lock:
                self.model = FisheyeModel.from_dict(doc['model'])
                self.model_mtime = mtime
                self.calibration_doc = doc
            self.pub_calibration.publish(String(data=json.dumps(doc)))
            fit = doc.get('fit', {})
            self.get_logger().info(f"loaded calibration: {fit.get('n_stars')} stars, "
                                   f"rms {fit.get('rms_deg', float('nan')):.3f} deg")
        except Exception as e:
            self.get_logger().error(f'cannot load calibration {path}: {e}')

    def model_for(self, w: int, h: int) -> FisheyeModel | None:
        with self.lock:
            m = self.model
        if m is None:
            return None
        if w > 0 and h > 0 and (w, h) != (m.image_width, m.image_height):
            return m.scaled(w, h)
        return m

    def solver_params(self) -> SolverParams:
        p = self.p
        sp = SolverParams(fov_deg=p('fov_deg'), max_mag=p('max_mag'), min_el_deg=p('min_elevation_deg'),
                          fit_tangential=p('fit_tangential'))
        if p('image_circle_radius') > 0:
            sp.radius_px = p('image_circle_radius')
        return sp

    def load_mask(self):
        f = self.p('mask_file')
        if f and os.path.exists(f):
            return cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        return None

    # ------------------------------------------------------------------ topics
    def image_callback(self, msg: Image):
        if self.collect_n <= 0:
            return
        try:
            frame = self.br.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if frame.ndim == 3:
                code = cv2.COLOR_RGB2GRAY if msg.encoding.lower().startswith('rgb') else cv2.COLOR_BGR2GRAY
                frame = cv2.cvtColor(frame, code)
            self.collected.append(frame.copy())
            st = msg.header.stamp
            self.collected_stamps.append(st.sec + st.nanosec * 1e-9)
            self.collect_n -= 1
            if self.collect_n <= 0:
                self.collect_done.set()
        except Exception as e:
            self.get_logger().error(f'frame conversion failed: {e}')

    def tracking_callback(self, msg: Tracking):
        if self.pub_tracks.get_subscription_count() == 0 or not msg.detections:
            return
        m = self.model_for(self.p('tracking_image_width'), self.p('tracking_image_height'))
        if m is None:
            return
        cx = np.array([d.bbox.x + d.bbox.width / 2.0 for d in msg.detections])
        cy = np.array([d.bbox.y + d.bbox.height / 2.0 for d in msg.detections])
        ex = np.array([d.bbox.x + d.bbox.width for d in msg.detections], dtype=float)
        az, el = m.pixel_to_azel(cx, cy)
        az2, el2 = m.pixel_to_azel(ex, cy)
        half = angular_separation_deg(az, el, az2, el2)
        out = TrackAzElArray()
        out.header = msg.header
        for d, x, y, a, e, hw in zip(msg.detections, cx, cy, az, el, half):
            out.tracks.append(TrackAzEl(id=d.id, state=d.state, x=float(x), y=float(y),
                                        azimuth_deg=float(a), elevation_deg=float(e),
                                        angular_width_deg=float(2 * hw)))
        self.pub_tracks.publish(out)

    # ------------------------------------------------------------------ capture
    def grab_stack(self):
        n = max(1, int(self.p('stack_frames')))
        self.collected, self.collected_stamps = [], []
        self.collect_done.clear()
        self.collect_n = n
        if not self.collect_done.wait(float(self.p('capture_timeout_s'))):
            self.collect_n = 0
            if not self.collected:
                raise RuntimeError(f"no frames received on {self.p('image_topic')}")
        frames, stamps = self.collected, self.collected_stamps
        stacked = stack_frames(frames)
        t_mid = float(np.median(stamps))
        return stacked, len(frames), t_mid

    def do_capture(self, utc_time: str, calibrate: bool):
        if not self.busy.acquire(blocking=False):
            raise RuntimeError('a capture/calibration is already running')
        try:
            stacked, n, t_mid = self.grab_stack()
            if utc_time:
                t = parse_time(utc_time)
            elif self.p('time_override'):
                t = parse_time(self.p('time_override'))
            else:
                t = datetime.fromtimestamp(t_mid, tz=timezone.utc)
            t = t + timedelta(seconds=float(self.p('time_offset_s')))
            img = np.clip(stacked, 0, 255).astype(np.uint8) if stacked.max() <= 255 \
                else np.clip(stacked, 0, 65535).astype(np.uint16)
            conditions = self.sky.conditions(t)
            conditions.pop('clear_dark_windows', None)
            cap = self.store.new_capture(img, t, self.site(), self.load_mask(),
                                         DetectorParams(snr=self.p('detection_snr')), frames_stacked=n,
                                         conditions=conditions)
            self.store.write_overlay(cap, self.model_for(0, 0))
            self.get_logger().info(f'capture {cap.capture_id}: {len(cap.detections)} stars detected '
                                   f'({n} frames stacked, sun {cap.sun_elevation_deg:.1f} deg)')
            res = None
            if calibrate:
                res = self.store.calibrate(sp=self.solver_params())
                self.reload_model()
                cap = self.store.load_capture(cap.capture_id)
            return cap, res
        finally:
            self.busy.release()

    def capture_callback(self, request, response):
        try:
            cap, res = self.do_capture(request.utc_time, request.calibrate)
            response.success = True
            response.capture_id = cap.capture_id
            response.num_detections = len(cap.detections)
            response.num_matched = len(cap.matches) + len(cap.manual)
            response.rms_deg = float(res.rms_deg) if res else float('nan')
            cond = cap.conditions or {}
            fc = (cond.get('forecast') or {}).get('cloud_cover')
            warn = f' (warning: forecast {fc:.0f}% cloud)' if fc is not None and \
                fc > self.p('weather_max_cloud_cover_pct') else ''
            response.message = ('ok' + warn) if res is None else \
                f'calibrated: {res.n_matches} stars, rms {res.rms_deg:.3f} deg ({res.quality})' + warn
        except Exception as e:
            self.get_logger().error(f'capture failed: {e}\n{tb.format_exc()}')
            response.success = False
            response.message = str(e)
        return response

    def auto_capture(self):
        if self.busy.locked():
            return
        ok, reason = self.sky.should_capture()
        if not ok:
            self.get_logger().debug(f'auto capture skipped: {reason}')
            return
        try:
            self.get_logger().info(f'auto capture: {reason}')
            cap, res = self.do_capture('', bool(self.p('auto_calibrate')))
            if res:
                self.get_logger().info(f'auto calibration: {res.n_matches} stars, rms {res.rms_deg:.3f} deg')
        except Exception as e:
            self.get_logger().warning(f'auto capture failed: {e}')

    def publish_outlook(self):
        if not self.p('weather_enabled'):
            return
        if not self.sky.refresh():
            self.get_logger().warning(f'weather forecast unavailable: {self.sky.last_error}')
        try:
            out = self.sky.conditions()
            self.pub_outlook.publish(String(data=json.dumps(out)))
            wins = out.get('clear_dark_windows') or []
            if wins:
                w = wins[0]
                self.get_logger().info(f"next clear dark window: {w['start_utc']} -> {w['end_utc']} "
                                       f"({w['mean_cloud_cover']}% cloud)")
            elif self.sky.forecast is not None:
                self.get_logger().info('no clear dark window forecast in the next 36 h')
        except Exception as e:
            self.get_logger().error(f'outlook failed: {e}')

    # ------------------------------------------------------------------ calibrate
    def calibrate_callback(self, request, response):
        if not self.busy.acquire(blocking=False):
            response.success = False
            response.message = 'a capture/calibration is already running'
            return response
        try:
            res = self.store.calibrate(list(request.capture_ids) or None, self.solver_params(),
                                       force=request.force)
            self.reload_model()
            response.success = True
            response.num_stars = res.n_matches
            response.rms_px = res.rms_px
            response.rms_deg = res.rms_deg
            response.quality = res.quality
            response.calibration_json = json.dumps(self.store.load_calibration())
            response.message = f'{res.n_matches} stars, rms {res.rms_deg:.3f} deg ({res.quality})'
        except Exception as e:
            self.get_logger().error(f'calibration failed: {e}')
            response.success = False
            response.message = str(e)
        finally:
            self.busy.release()
        return response

    # ------------------------------------------------------------------ conversions
    def pixel_to_azel_callback(self, request, response):
        m = self.model_for(request.image_width, request.image_height)
        if m is None:
            response.success, response.message = False, 'not calibrated yet'
            return response
        if len(request.x) != len(request.y):
            response.success, response.message = False, 'x and y differ in length'
            return response
        az, el = m.pixel_to_azel(np.array(request.x), np.array(request.y))
        response.azimuth_deg = [float(a) for a in az]
        response.elevation_deg = [float(e) for e in el]
        response.extrapolated = [bool(b) for b in m.extrapolated(np.array(request.x), np.array(request.y))]
        response.success, response.message = True, 'ok'
        return response

    def azel_to_pixel_callback(self, request, response):
        m = self.model_for(request.image_width, request.image_height)
        if m is None:
            response.success, response.message = False, 'not calibrated yet'
            return response
        if len(request.azimuth_deg) != len(request.elevation_deg):
            response.success, response.message = False, 'azimuth and elevation differ in length'
            return response
        u, v, vis = m.azel_to_pixel(np.array(request.azimuth_deg), np.array(request.elevation_deg))
        response.x = [float(a) for a in u]
        response.y = [float(b) for b in v]
        response.in_image = [bool(b) for b in vis]
        response.success, response.message = True, 'ok'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = StarTrackerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
