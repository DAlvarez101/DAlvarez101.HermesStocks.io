from dataclasses import dataclass


@dataclass
class Station:
    icao: str
    lat: float
    lon: float
    elevation_ft: float
    role: str


STATIONS = [
    Station("KDFW", 32.897, -97.038, 607, "airport_southwest"),
    Station("KDAL", 32.848, -96.851, 487, "target"),
    Station("KADS", 33.075, -96.837, 645, "north_suburban"),
    Station("KAFW", 32.990, -97.319, 679, "northwest_exurban"),
    Station("KDTO", 33.200, -97.198, 642, "north_rural"),
    Station("KGKY", 32.664, -97.094, 628, "south_arlington"),
    Station("KACT", 31.611, -97.230, 686, "south_rural"),
    Station("KTYR", 32.354, -95.402, 550, "east_rural"),
]

TARGET_ICAO = "KDAL"
CACHE_DIR = "data/cache"
