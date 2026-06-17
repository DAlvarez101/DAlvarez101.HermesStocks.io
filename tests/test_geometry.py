import math

import pandas as pd
import pytest

from dfw_temp_model.config import STATIONS, Station
from dfw_temp_model.features.geometry import (
    bearing_deg,
    distance_m,
    local_xy,
    make_projection,
    smallest_angle_diff,
    station_geometry_table,
)


def test_make_projection_returns_pyproj_proj():
    proj = make_projection(32.897, -97.038)
    assert proj is not None
    # pyproj.Proj is callable and transforms lat/lon -> x/y in meters for aeqd
    x, y = proj(-97.038, 32.897)
    assert abs(x) < 1.0
    assert abs(y) < 1.0


def test_local_coordinates_target_maps_to_origin():
    lat0, lon0 = 32.897, -97.038
    x, y = local_xy(lat0, lon0, lat0, lon0)
    assert abs(x) < 0.1
    assert abs(y) < 0.1


def test_local_coordinates_east_is_positive_x():
    # A point slightly east should have positive x and near-zero y
    x, y = local_xy(32.897, -97.0, 32.897, -97.038)
    assert x > 1000
    assert abs(y) < 100


def test_distance_and_bearing():
    # KDFW -> KDAL: ~20 km east-southeast, bearing ~100°
    x1, y1 = local_xy(32.897, -97.038, 32.897, -97.038)
    x2, y2 = local_xy(32.848, -96.851, 32.897, -97.038)
    d = distance_m(x1, y1, x2, y2)
    b = bearing_deg(x1, y1, x2, y2)
    assert 18000 < d < 22000
    assert 90 < b < 110


def test_bearing_north():
    assert abs(bearing_deg(0, 0, 0, 1000) - 0.0) < 1e-9


def test_bearing_east():
    assert abs(bearing_deg(0, 0, 1000, 0) - 90.0) < 1e-9


def test_bearing_south():
    assert abs(bearing_deg(0, 0, 0, -1000) - 180.0) < 1e-9


def test_bearing_west():
    assert abs(bearing_deg(0, 0, -1000, 0) - 270.0) < 1e-9


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (10, 350, 20),
        (350, 10, 20),
        (0, 180, 180),
        (90, 270, 180),
        (5, 5, 0),
        (-90, 90, 180),
        (720, 0, 0),
    ],
)
def test_smallest_angle_diff(a, b, expected):
    assert abs(smallest_angle_diff(a, b) - expected) < 1e-9


def test_station_geometry_table():
    table = station_geometry_table(STATIONS)
    assert isinstance(table, pd.DataFrame)
    expected_cols = {
        "x_m",
        "y_m",
        "dist_m",
        "dist_km",
        "bearing_from_target_deg",
        "elevation_diff_ft",
        "role",
    }
    assert expected_cols.issubset(set(table.columns))

    kdal = table.loc["KDAL"]
    assert abs(kdal["dist_km"]) < 0.01
    assert abs(kdal["bearing_from_target_deg"]) < 1.0 or abs(
        kdal["bearing_from_target_deg"] - 360
    ) < 1.0
    assert kdal["elevation_diff_ft"] == 0

    # Neighbor distances should be positive
    neighbors = table[table.index != "KDAL"]
    assert (neighbors["dist_km"] > 0).all()


def test_station_geometry_table_custom_target():
    stations = [
        Station("KDAL", 32.0, -97.0, 500, "target"),
        Station("NORTH", 33.0, -97.0, 600, "north"),
    ]
    table = station_geometry_table(stations)
    assert table.loc["KDAL", "dist_km"] < 0.01
    north = table.loc["NORTH"]
    assert north["dist_km"] > 100
    assert 0 <= north["bearing_from_target_deg"] < 1
    assert north["elevation_diff_ft"] == 100
