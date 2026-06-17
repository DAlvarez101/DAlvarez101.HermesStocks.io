from dfw_temp_model.config import STATIONS, TARGET_ICAO


def test_station_count():
    assert len(STATIONS) == 8


def test_target_icao():
    assert TARGET_ICAO == "KDAL"


def test_exactly_one_target_and_is_kdal():
    targets = [s for s in STATIONS if s.role == "target"]
    assert len(targets) == 1
    assert targets[0].icao == "KDAL"