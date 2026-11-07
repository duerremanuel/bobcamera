"""Star tracker calibration for Bob all-sky fisheye cameras.

Detects stars in night frames, identifies them against a catalog using the
camera site and capture time, and fits a fisheye model that converts image
pixels to true-north azimuth / elevation (and back)."""

from .fisheye_model import FisheyeModel, azel_to_enu, enu_to_azel, angular_separation_deg
from .star_catalog import Site, StarCatalog
from .star_detector import detect_stars, DetectorParams, StarDetection, stack_frames
from .star_matcher import solve, SolverParams, SolveResult, StarMatch
from .calibration_store import CalibrationStore, Capture, parse_time
