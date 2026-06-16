"""Local Cartesian geometry utilities for DFW-area stations.

Uses an azimuthal equidistant (aeqd) projection centered on the target station
(KDFW by default) so that distances and bearings in meters/degrees are locally
accurate.
"""

from __future__ import annotations

import math

import pandas as pd
import pyproj

from dfw_temp_model.config import TARGET_ICAO, Station


def make_projection(lat0: float, lon0: float) -> pyproj.Proj:
    """Return a pyproj aeqd projection centered on (lat0, lon0)."""
    return pyproj.Proj(proj="aeqd", lat_0=lat0, lon_0=lon0, units="m")


def local_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Convert (lat, lon) to meters relative to (lat0, lon0)."""
    proj = make_projection(lat0, lon0)
    x, y = proj(lon, lat)
    return float(x), float(y)


def distance_m(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points in meters."""
    return math.hypot(x2 - x1, y2 - y1)


def bearing_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Bearing from (x1, y1) to (x2, y2), 0° = north, clockwise.

    Uses the math convention where +x is east and +y is north.
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return 0.0
    bearing = math.degrees(math.atan2(dx, dy))
    if bearing < 0:
        bearing += 360.0
    return bearing


def smallest_angle_diff(a: float, b: float) -> float:
    """Absolute smallest angle difference between a and b, in degrees."""
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return abs(diff)


def station_geometry_table(stations: list[Station]) -> pd.DataFrame:
    """Build a DataFrame with local geometry relative to the target station."""
    target = next((s for s in stations if s.icao == TARGET_ICAO), None)
    if target is None:
        raise ValueError(f"Target station {TARGET_ICAO!r} not found in station list")

    lat0, lon0 = target.lat, target.lon
    rows = []
    for s in stations:
        x, y = local_xy(s.lat, s.lon, lat0, lon0)
        dist = distance_m(0.0, 0.0, x, y)
        bearing = bearing_deg(0.0, 0.0, x, y)
        rows.append(
            {
                "icao": s.icao,
                "x_m": x,
                "y_m": y,
                "dist_m": dist,
                "dist_km": dist / 1000.0,
                "bearing_from_target_deg": bearing,
                "elevation_diff_ft": s.elevation_ft - target.elevation_ft,
                "role": s.role,
            }
        )

    df = pd.DataFrame(rows).set_index("icao")
    return df
