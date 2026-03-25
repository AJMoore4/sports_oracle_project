"""
sports_oracle/utils/geo.py

Travel distance and altitude calculations for tournament games.

TRAVEL DISTANCE:
  Haversine formula between team home city and venue city.
  Doesn't need an API — we store lat/lng for all D1 team
  home cities and common tournament venues.

ALTITUDE:
  Higher altitude = slightly lower shooting percentages,
  faster fatigue. Effect is small but measurable for teams
  coming from sea level to play in Denver, Salt Lake City, etc.

USAGE:
    geo = GeoLookup()
    dist = geo.travel_distance("Duke", "State Farm Stadium")
    alt = geo.altitude_diff("Duke", "State Farm Stadium")
"""

from __future__ import annotations
import math
import logging
from typing import Optional

logger = logging.getLogger("sports_oracle.geo")


# ── Team home locations (lat, lng, altitude_ft) ──────────────────────────────
# Covers all teams likely to appear in the NCAA tournament.
# Altitude in feet above sea level.
# Only need tournament-caliber teams + a reasonable default for unknowns.

TEAM_LOCATIONS: dict[str, tuple[float, float, int]] = {
    # ── Power conferences + frequent tournament teams ──
    # (lat, lng, altitude_ft)

    # ACC
    "Boston College":     (42.3355, -71.1685, 80),
    "California":         (37.8716, -122.2590, 177),
    "Clemson":            (34.6783, -82.8322, 850),
    "Duke":               (36.0014, -78.9382, 400),
    "Florida St.":        (30.4419, -84.2985, 200),
    "Georgia Tech":       (33.7813, -84.3926, 1050),
    "Louisville":         (38.2169, -85.7588, 460),
    "Miami (FL)":         (25.7136, -80.2713, 10),
    "NC State":           (35.7872, -78.6705, 430),
    "North Carolina":     (35.9049, -79.0469, 500),
    "Notre Dame":         (41.7002, -86.2353, 720),
    "Pittsburgh":         (40.4443, -79.9608, 1225),
    "SMU":                (32.8412, -96.7836, 485),
    "Stanford":           (37.4316, -122.1700, 100),
    "Syracuse":           (43.0369, -76.1365, 410),
    "Virginia":           (38.0336, -78.5080, 480),
    "Virginia Tech":      (37.2209, -80.4253, 2080),
    "Wake Forest":        (36.1343, -80.2773, 940),

    # Big 12
    "Arizona":            (32.2317, -110.9533, 2410),
    "Arizona St.":        (33.4242, -111.9281, 1135),
    "Baylor":             (31.5583, -97.1143, 470),
    "BYU":                (40.2518, -111.6493, 4550),
    "Cincinnati":         (39.1313, -84.5150, 480),
    "Colorado":           (40.0076, -105.2659, 5430),
    "Houston":            (29.7199, -95.3422, 50),
    "Iowa St.":           (42.0140, -93.6358, 940),
    "Kansas":             (38.9543, -95.2558, 860),
    "Kansas St.":         (39.1836, -96.5717, 1060),
    "Oklahoma St.":       (36.1216, -97.0692, 895),
    "TCU":                (32.7096, -97.3630, 650),
    "Texas":              (30.2849, -97.7341, 500),
    "Texas Tech":         (33.5843, -101.8453, 3280),
    "UCF":                (28.6024, -81.2001, 80),
    "Utah":               (40.7649, -111.8421, 4780),
    "West Virginia":      (39.6480, -79.9559, 950),

    # Big East
    "Butler":             (39.8390, -86.1691, 720),
    "Connecticut":        (41.8065, -72.2539, 640),
    "Creighton":          (41.2647, -95.9445, 1090),
    "DePaul":             (41.9252, -87.6538, 595),
    "Georgetown":         (38.9076, -77.0723, 70),
    "Marquette":          (43.0389, -87.9365, 635),
    "Providence":         (41.8403, -71.4350, 50),
    "Seton Hall":         (40.7424, -74.2370, 100),
    "St. John's":         (40.7262, -73.7944, 50),
    "Villanova":          (40.0343, -75.3372, 405),
    "Xavier":             (39.1491, -84.4736, 550),

    # Big Ten
    "Illinois":           (40.1020, -88.2272, 740),
    "Indiana":            (39.1682, -86.5230, 780),
    "Iowa":               (41.6611, -91.5302, 700),
    "Maryland":           (38.9860, -76.9440, 150),
    "Michigan":           (42.2681, -83.7486, 840),
    "Michigan St.":       (42.7284, -84.4816, 850),
    "Minnesota":          (44.9740, -93.2277, 840),
    "Nebraska":           (40.8202, -96.7005, 1180),
    "Northwestern":       (42.0565, -87.6753, 610),
    "Ohio St.":           (39.9985, -83.0152, 780),
    "Oregon":             (44.0448, -123.0726, 430),
    "Penn St.":           (40.8003, -77.8619, 1170),
    "Purdue":             (40.4237, -86.9212, 620),
    "Rutgers":            (40.5008, -74.4474, 100),
    "UCLA":               (34.0709, -118.4462, 420),
    "USC":                (34.0224, -118.2851, 180),
    "Washington":         (47.6567, -122.3066, 50),
    "Wisconsin":          (43.0731, -89.4012, 870),

    # SEC
    "Alabama":            (33.2098, -87.5692, 220),
    "Arkansas":           (36.0680, -94.1748, 1250),
    "Auburn":             (32.6010, -85.4877, 690),
    "Florida":            (29.6499, -82.3486, 100),
    "Georgia":            (33.9480, -83.3773, 760),
    "Kentucky":           (38.0306, -84.5040, 980),
    "LSU":                (30.4133, -91.1832, 55),
    "Mississippi St.":    (33.4552, -88.7882, 200),
    "Missouri":           (38.9404, -92.3277, 770),
    "Oklahoma":           (35.2058, -97.4457, 1200),
    "Ole Miss":           (34.3655, -89.5344, 470),
    "South Carolina":     (34.0007, -81.0348, 300),
    "Tennessee":          (35.9544, -83.9295, 900),
    "Texas A&M":          (30.6174, -96.3403, 310),
    "Vanderbilt":         (36.1445, -86.8027, 550),

    # Mid-majors with tournament history
    "Gonzaga":            (47.6673, -117.4015, 1920),
    "Saint Mary's":       (37.8402, -122.1110, 290),
    "Memphis":            (35.1174, -89.9711, 300),
    "Dayton":             (39.7400, -84.1793, 740),
    "San Diego St.":      (32.7757, -117.0719, 390),
    "VCU":                (37.5485, -77.4533, 160),
    "Wichita St.":        (37.7194, -97.2950, 1300),
    "Nevada":             (39.5441, -119.8138, 4505),
    "New Mexico":         (35.0844, -106.6504, 5312),
    "Boise St.":          (43.6036, -116.2025, 2710),
    "Drake":              (41.6031, -93.6540, 820),
    "Furman":             (34.9260, -82.4400, 1000),
    "Princeton":          (40.3431, -74.6551, 50),
    "Oral Roberts":       (36.0551, -95.9773, 700),
    "Loyola Chicago":     (41.9992, -87.6577, 595),
    "Grand Canyon":       (33.5103, -112.0997, 1100),
    "Colgate":            (42.8193, -75.5263, 1100),
    "Fairleigh Dickinson": (40.7592, -74.0320, 20),
    "Saint Peter's":      (40.7468, -74.0506, 10),
    "FGCU":               (26.4615, -81.7729, 10),
    "Florida Atlantic":   (26.3713, -80.1016, 15),
    "Iona":               (40.9303, -73.8328, 120),
    "Belmont":            (36.1356, -86.7989, 550),
    "Murray St.":         (36.6131, -88.3148, 540),
    "Vermont":            (44.4759, -73.2121, 200),
    "Wagner":             (40.5843, -74.0973, 80),
    "Yale":               (41.3163, -72.9223, 25),
    "Liberty":            (37.3519, -79.1834, 930),
    "James Madison":      (38.4376, -78.8697, 1350),
    "Kennesaw St.":       (34.0378, -84.5812, 1090),
    "Samford":            (33.4637, -86.7917, 620),
    "Duquesne":           (40.4354, -79.9948, 1200),
    "McNeese":            (30.2066, -93.2185, 15),
    "Stetson":            (29.0553, -81.3029, 30),
    "Oakland":            (42.6741, -83.2187, 1000),
    "Longwood":           (37.2990, -78.3949, 500),
    "Colorado St.":       (40.5734, -105.0865, 4980),
}

# ── Common NCAA tournament venues ─────────────────────────────────────────────
# (lat, lng, altitude_ft)
VENUE_LOCATIONS: dict[str, tuple[float, float, int]] = {
    # First/Second Round sites (rotate yearly)
    "State Farm Stadium":           (33.5276, -112.2626, 1100),  # Glendale AZ
    "PPG Paints Arena":             (40.4395, -79.9890, 1200),   # Pittsburgh
    "Wells Fargo Center":           (39.9012, -75.1720, 30),     # Philadelphia
    "Nationwide Arena":             (39.9691, -83.0061, 770),     # Columbus
    "Gainbridge Fieldhouse":        (39.7640, -86.1555, 720),    # Indianapolis
    "United Center":                (41.8807, -87.6742, 595),    # Chicago
    "American Airlines Center":     (32.7905, -96.8103, 485),    # Dallas
    "Toyota Center":                (29.7508, -95.3621, 50),     # Houston
    "Ball Arena":                   (39.7487, -105.0077, 5280),  # Denver
    "Delta Center":                 (40.7683, -111.9011, 4226),  # Salt Lake City
    "Chase Center":                 (37.7680, -122.3877, 5),     # San Francisco
    "T-Mobile Arena":               (36.1028, -115.1784, 2030),  # Las Vegas
    "Crypto.com Arena":             (34.0430, -118.2673, 300),   # LA
    "KFC Yum! Center":              (38.2572, -85.7585, 460),    # Louisville
    "Spectrum Center":              (35.2251, -80.8392, 750),    # Charlotte
    "Barclays Center":              (40.6828, -73.9758, 30),     # Brooklyn
    "Bon Secours Wellness Arena":   (34.8598, -82.3988, 1000),   # Greenville SC
    "KeyBank Center":               (42.8750, -78.8764, 585),    # Buffalo
    "Spokane Arena":                (47.6660, -117.4035, 1920),  # Spokane
    "Legacy Arena":                 (33.5211, -86.8142, 620),    # Birmingham
    "MVP Arena":                    (42.6525, -73.7594, 20),     # Albany

    # Regional / Sweet 16 / Elite 8 sites
    "Madison Square Garden":        (40.7505, -73.9934, 30),     # New York
    "TD Garden":                    (42.3662, -71.0621, 20),     # Boston
    "FedExForum":                   (35.1381, -90.0505, 300),    # Memphis
    "Dickies Arena":                (32.7362, -97.3565, 650),    # Fort Worth
    "State Farm Arena":             (33.7573, -84.3963, 1050),   # Atlanta
    "Lucas Oil Stadium":            (39.7601, -86.1639, 720),    # Indianapolis
    "AT&T Stadium":                 (32.7473, -97.0945, 580),    # Arlington TX
    "Alamodome":                    (29.4171, -98.4879, 650),    # San Antonio
    "NRG Stadium":                  (29.6847, -95.4107, 50),     # Houston
    "U.S. Bank Stadium":            (44.9736, -93.2575, 840),    # Minneapolis
    "Caesars Superdome":            (29.9511, -90.0812, 3),      # New Orleans
    "University of Phoenix Stadium": (33.5276, -112.2626, 1100), # Glendale AZ

    # Final Four sites
    "NRG Stadium (Final Four)":     (29.6847, -95.4107, 50),
    "Alamodome (Final Four)":       (29.4171, -98.4879, 650),
    "Lucas Oil Stadium (Final Four)": (39.7601, -86.1639, 720),
}

# Default location for unknown teams/venues
_DEFAULT_LOCATION = (39.8283, -98.5795, 2000)  # Geographic center of US


class GeoLookup:
    """
    Computes travel distance and altitude differential
    between a team's home and a tournament venue.
    """

    def __init__(self):
        self._team_cache = TEAM_LOCATIONS.copy()
        self._venue_cache = VENUE_LOCATIONS.copy()

    def get_team_location(self, team: str) -> tuple[float, float, int]:
        """Get (lat, lng, altitude_ft) for a team. Returns default if unknown."""
        loc = self._team_cache.get(team)
        if loc:
            return loc

        # Try case-insensitive
        for name, loc in self._team_cache.items():
            if name.lower() == team.lower():
                return loc

        logger.debug(f"No location data for team '{team}', using US center")
        return _DEFAULT_LOCATION

    def get_venue_location(
        self,
        venue_name: Optional[str] = None,
        venue_city: Optional[str] = None,
        venue_state: Optional[str] = None,
    ) -> tuple[float, float, int]:
        """Get (lat, lng, altitude_ft) for a venue."""
        if venue_name:
            loc = self._venue_cache.get(venue_name)
            if loc:
                return loc

            # Partial match
            name_lower = venue_name.lower()
            for name, loc in self._venue_cache.items():
                if name_lower in name.lower() or name.lower() in name_lower:
                    return loc

        logger.debug(
            f"No location for venue '{venue_name}', using default"
        )
        return _DEFAULT_LOCATION

    def travel_distance(
        self,
        team: str,
        venue_name: Optional[str] = None,
        venue_lat: Optional[float] = None,
        venue_lng: Optional[float] = None,
    ) -> float:
        """
        Calculate travel distance in miles from team home to venue.
        Uses Haversine formula (great-circle distance).

        Returns distance in miles.
        """
        team_lat, team_lng, _ = self.get_team_location(team)

        if venue_lat is not None and venue_lng is not None:
            v_lat, v_lng = venue_lat, venue_lng
        else:
            v_lat, v_lng, _ = self.get_venue_location(venue_name)

        return self._haversine(team_lat, team_lng, v_lat, v_lng)

    def altitude_diff(
        self,
        team: str,
        venue_name: Optional[str] = None,
    ) -> int:
        """
        Calculate altitude difference in feet.
        Positive = venue is higher than team's home.
        """
        _, _, team_alt = self.get_team_location(team)
        _, _, venue_alt = self.get_venue_location(venue_name)
        return venue_alt - team_alt

    def travel_context(
        self,
        team: str,
        venue_name: Optional[str] = None,
        venue_lat: Optional[float] = None,
        venue_lng: Optional[float] = None,
    ) -> dict:
        """
        Full travel context for a team playing at a venue.
        Returns dict ready for the prediction engine.
        """
        distance = self.travel_distance(
            team, venue_name, venue_lat, venue_lng
        )
        alt_diff = self.altitude_diff(team, venue_name)

        return {
            "team": team,
            "travel_distance_miles": round(distance, 1),
            "altitude_diff_ft": alt_diff,
            "is_long_travel": distance > 1000,
            "is_high_altitude": alt_diff > 3000,
        }

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine formula — returns distance in miles."""
        R = 3959  # Earth radius in miles

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))

        return R * c


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    geo = GeoLookup()

    print("\n🌎 GeoLookup — Travel Distance Tests")
    print("=" * 55)

    tests = [
        ("Duke", "State Farm Stadium"),
        ("Duke", "Madison Square Garden"),
        ("Gonzaga", "Caesars Superdome"),
        ("Kansas", "Gainbridge Fieldhouse"),
        ("Colorado", "Ball Arena"),         # ~25 miles, home game
        ("Florida", "Delta Center"),        # sea level → 4200ft
    ]

    for team, venue in tests:
        ctx = geo.travel_context(team, venue)
        print(
            f"  {team:20s} → {venue:30s} "
            f"| {ctx['travel_distance_miles']:7.1f} mi "
            f"| alt Δ {ctx['altitude_diff_ft']:+5d} ft"
        )
