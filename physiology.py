"""
Physiology & weather helpers.

Everything the model needs that is *not* learned from data lives here:

  * pace  <-> velocity  <-> log-velocity conversions,
  * the fixed weather -> "effective temperature" mapping,
  * the heat-penalty basis functions,
  * the physiological link between threshold pace and vVO2max pace.

We model performance in *log-velocity* (natural log of m/s).  Why:

  * Fitness and the weather penalty act *multiplicatively* on velocity
    (e.g. "5% slower in the heat").  Multiplicative effects become
    additive in log-space, which makes the whole model linear-Gaussian
    and therefore exactly solvable (see model.py).
  * Runners think in pace (min/km), so we only convert to pace for display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as dtdate, datetime, time as dttime, timezone
from functools import lru_cache

try:
    from zoneinfo import ZoneInfo
    _HAVE_ZONEINFO = True
except Exception:  # pragma: no cover
    _HAVE_ZONEINFO = False


@lru_cache(maxsize=None)
def _zone(tz_name: str):
    """IANA timezone name -> tzinfo, falling back to UTC if unavailable.

    The `tzdata` package (in requirements) provides the database on systems
    that lack one, so all the reference cities resolve in the cloud too.
    """
    if _HAVE_ZONEINFO:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    return timezone.utc

# --------------------------------------------------------------------------
# Pace <-> velocity
# --------------------------------------------------------------------------

def parse_pace(text: str) -> float:
    """Parse a pace string into seconds per km.

    Accepts "m:ss", "mm:ss", or a plain number of seconds.
    """
    text = str(text).strip()
    if ":" in text:
        mm, ss = text.split(":")
        return int(mm) * 60 + float(ss)
    return float(text)


def format_pace(sec_per_km: float) -> str:
    """Seconds per km -> 'm:ss' string."""
    if sec_per_km is None or not math.isfinite(sec_per_km) or sec_per_km <= 0:
        return "--:--"
    sec_per_km = round(sec_per_km)
    m, s = divmod(int(sec_per_km), 60)
    return f"{m}:{s:02d}"


def pace_to_velocity(sec_per_km: float) -> float:
    """Seconds per km -> velocity in m/s."""
    return 1000.0 / sec_per_km


def velocity_to_pace(v_ms: float) -> float:
    """Velocity in m/s -> seconds per km."""
    return 1000.0 / v_ms


def pace_to_logv(sec_per_km: float) -> float:
    return math.log(pace_to_velocity(sec_per_km))


def logv_to_pace(logv: float) -> float:
    return velocity_to_pace(math.exp(logv))


# --------------------------------------------------------------------------
# Threshold  <->  vVO2max
# --------------------------------------------------------------------------
# We treat vVO2max (velocity at VO2max, ~3-5 min interval velocity) as the
# single "fitness anchor".  Lactate-threshold velocity is a fairly stable
# fraction of it.  Literature puts vLT at ~85-92% of vVO2max; we default to
# 0.90.  This is what lets the app predict *both* paces from sessions of
# *either* type: every session is an observation of the same latent fitness,
# just offset by log(LT_FRACTION) for threshold sessions.
#
# The offset is also re-estimated per user once both session types exist
# (the `beta_r` parameter in the model), so the default only matters early on.

DEFAULT_LT_FRACTION = 0.90


# --------------------------------------------------------------------------
# Weather  ->  effective temperature  ->  heat penalty
# --------------------------------------------------------------------------
# The dominant, best-established environmental effect on endurance running is
# heat.  Sun, humidity and rain mostly act by changing the *effective* thermal
# load, so we fold them into a single "effective temperature" with fixed,
# physiologically-sensible adjustments, and let the model learn only the
# overall heat *sensitivity* (two coefficients) per user.  Keeping the number
# of learned weather parameters tiny is what makes this work on ~50 sessions.

THERMAL_OPTIMUM_C = 12.0  # ~ideal racing temperature for distance running

SKY_OPTIONS = ["sunny", "partly cloudy", "overcast"]
RAIN_OPTIONS = ["none", "light", "heavy"]

RAIN_ADJUST = {"none": 0.0, "light": -1.0, "heavy": -2.0}  # rain cools
HUMIDITY_REF = 50.0          # % RH considered "neutral"
HUMIDITY_PER_10PCT = 1.0     # +1 C effective per 10% RH above the reference

# --- Solar radiation -------------------------------------------------------
# Air temperature ignores radiant heat, but direct sun is a big part of why a
# noon run feels harder than an evening run at the same temperature.  We
# compute the sun's elevation above the horizon (deterministic from location +
# date + clock time) and turn it into an extra "effective temperature" load,
# scaled down by cloud cover.  The location is the user's chosen city.

K_SOLAR = 6.0                # max effective-temp add (degC) at full sun, high noon
CLOUD_TRANSMISSION = {"sunny": 1.0, "partly cloudy": 0.5, "overcast": 0.15}
# Used when a session has no recorded time of day (mid value ~ old fixed bump).
LEGACY_SOLAR_FACTOR = 0.5


@dataclass(frozen=True)
class Location:
    """A running location: display name + coordinates + IANA timezone."""
    name: str
    lat: float           # degrees north
    lon: float           # degrees east
    tz: str              # IANA timezone name, e.g. "Europe/Berlin"


DEFAULT_LOCATION = Location("Stuttgart, Germany", 48.78, 9.18, "Europe/Berlin")

# A spread of reference cities; on sign-up a user picks the one closest to where
# they run.  Latitude + longitude + timezone are all we need for the sun model.
# Europe-weighted, with a few on every other continent.
CITIES = [
    DEFAULT_LOCATION,
    Location("London, United Kingdom", 51.51, -0.13, "Europe/London"),
    Location("Paris, France", 48.85, 2.35, "Europe/Paris"),
    Location("Madrid, Spain", 40.42, -3.70, "Europe/Madrid"),
    Location("Stockholm, Sweden", 59.33, 18.07, "Europe/Stockholm"),
    Location("New York, USA", 40.71, -74.01, "America/New_York"),
    Location("Los Angeles, USA", 34.05, -118.24, "America/Los_Angeles"),
    Location("São Paulo, Brazil", -23.55, -46.63, "America/Sao_Paulo"),
    Location("Cairo, Egypt", 30.04, 31.24, "Africa/Cairo"),
    Location("Cape Town, South Africa", -33.92, 18.42, "Africa/Johannesburg"),
    Location("Dubai, UAE", 25.20, 55.27, "Asia/Dubai"),
    Location("Mumbai, India", 19.08, 72.88, "Asia/Kolkata"),
    Location("Singapore", 1.35, 103.82, "Asia/Singapore"),
    Location("Tokyo, Japan", 35.68, 139.69, "Asia/Tokyo"),
    Location("Sydney, Australia", -33.87, 151.21, "Australia/Sydney"),
]


def solar_elevation(d: dtdate, t: dttime, loc: Location = DEFAULT_LOCATION) -> float:
    """Sun elevation angle (degrees above horizon) for a local clock time at
    `loc`.  Handles the location's DST automatically via its timezone.
    Negative = sun below the horizon (night)."""
    local = datetime(d.year, d.month, d.day, t.hour, t.minute, tzinfo=_zone(loc.tz))
    utc = local.astimezone(timezone.utc)
    n = utc.timetuple().tm_yday
    decl = 23.45 * math.sin(math.radians(360 / 365 * (n - 81)))
    b = math.radians(360 / 364 * (n - 81))
    eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)  # minutes
    utc_h = utc.hour + utc.minute / 60 + utc.second / 3600
    lst = utc_h + loc.lon / 15 + eot / 60            # local solar time (hours)
    hra = math.radians(15 * (lst - 12))               # hour angle
    phi, dec = math.radians(loc.lat), math.radians(decl)
    sin_elev = math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(hra)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))


def _raw_solar_load(elev_deg: float) -> float:
    """Clear-sky horizontal beam intensity (arbitrary units) at a given sun
    elevation, using a Kasten-Young air-mass attenuation."""
    if elev_deg <= 0:
        return 0.0
    s = math.sin(math.radians(elev_deg))
    air_mass = 1.0 / (s + 0.50572 * (elev_deg + 6.07995) ** -1.6364)
    dni = 0.7 ** (air_mass ** 0.678)   # transmitted fraction
    return dni * s


# Fixed reference: a clear summer-solstice solar noon at mid-latitude -> 1.0.
# Lower-latitude cities reach higher sun and saturate near 1.0 (more solar
# load); higher-latitude cities peak below 1.0 -- the latitude effect we want.
_SOLAR_NORM = _raw_solar_load(90 - (DEFAULT_LOCATION.lat - 23.45))


def solar_load_factor(elev_deg: float) -> float:
    """Normalised solar load in [0, 1]; 1.0 ~ clear summer-noon sun."""
    return min(1.0, _raw_solar_load(elev_deg) / _SOLAR_NORM)


def solar_bonus(w: "Weather") -> float:
    """Effective-temperature add-on (degC) from direct sun, given cloud cover.
    Falls back to a fixed mid-day estimate when no time of day is recorded."""
    trans = CLOUD_TRANSMISSION.get(w.sky, 0.15)
    if w.date is not None and w.time is not None:
        factor = solar_load_factor(solar_elevation(w.date, w.time, w.loc))
    else:
        factor = LEGACY_SOLAR_FACTOR
    return K_SOLAR * trans * factor


def parse_time(text) -> dttime | None:
    """Parse 'HH:MM' (or a datetime.time) into a time, or None."""
    if text is None or text == "":
        return None
    if isinstance(text, dttime):
        return text
    hh, mm = str(text).split(":")[:2]
    return dttime(int(hh), int(mm))


@dataclass
class Weather:
    temp_c: float
    sky: str = "overcast"
    rain: str = "none"
    humidity: float = 50.0
    date: dtdate | None = None    # needed for the solar term
    time: dttime | None = None
    loc: Location = DEFAULT_LOCATION   # where the run happens (sun geometry)

    @classmethod
    def from_row(cls, row: dict, loc: Location = DEFAULT_LOCATION) -> "Weather":
        return cls(
            temp_c=float(row["temp_c"]),
            sky=row.get("sky", "overcast"),
            rain=row.get("rain", "none"),
            humidity=float(row.get("humidity", 50.0) or 50.0),
            date=row.get("date"),
            time=parse_time(row.get("time")),
            loc=loc,
        )


def user_location(user: dict) -> Location:
    """Resolve a user row's stored city to a Location (Stuttgart if unset)."""
    try:
        lat, lon = user.get("lat"), user.get("lon")
        if lat is not None and lon is not None:
            return Location(user.get("city") or "—", float(lat), float(lon),
                            user.get("tz") or DEFAULT_LOCATION.tz)
    except (TypeError, ValueError, AttributeError):
        pass
    return DEFAULT_LOCATION


def effective_temperature(w: Weather) -> float:
    """Map raw weather to a single 'feels-like for running' temperature."""
    adj = solar_bonus(w)
    adj += RAIN_ADJUST.get(w.rain, 0.0)
    adj += HUMIDITY_PER_10PCT * (w.humidity - HUMIDITY_REF) / 10.0
    return w.temp_c + adj


def heat_penalty_basis(w: Weather) -> tuple[float, float]:
    """Return (P1, P2), the linear and quadratic heat-load basis functions.

    Both are zero at/below the thermal optimum and grow with effective
    temperature above it.  The model multiplies these by learned (negative)
    coefficients beta1, beta2 to produce the log-velocity slowdown, so the
    penalty is linear-in-parameters -> the model stays exactly solvable.
    """
    x = max(0.0, effective_temperature(w) - THERMAL_OPTIMUM_C)
    return x, x * x
