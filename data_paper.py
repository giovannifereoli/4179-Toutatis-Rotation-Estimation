"""
Numerical data taken directly from Takahashi, Busch & Scheeres (2013):
  * Table 2 - initial conditions / a-priori uncertainties of the state vector
              at the estimation epoch 17:49:47 UTC 1992 November 9.
  * Table 3 - observation log (3-1-3 Euler angles and instantaneous spin
              vectors from Goldstone/Arecibo delay-Doppler images, 1992-2008).

Units: Euler angles [deg], angular velocity [deg/day], inertia ratios [-],
COM-COF offset [km].
"""
import numpy as np

DAY_S = 86400.0

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GM_SUN_KM3_S2   = 1.32712440018e11        # km^3/s^2
GM_EARTH_KM3_S2 = 3.986004418e5           # km^3/s^2
GM_SUN   = GM_SUN_KM3_S2   * DAY_S**2      # km^3/day^2
GM_EARTH = GM_EARTH_KM3_S2 * DAY_S**2      # km^3/day^2

# Toutatis bulk parameters.  Only the first-degree (COM-COF) torque needs an
# absolute mass-to-Izz scale; the second-degree (inertia-ratio) dynamics are
# independent of it.  We derive M*/Izz from the paper's *own* quantities by
# adopting the uniform-density model the paper uses for its interior discussion
# (a uniform triaxial ellipsoid):
#     I_xx = (M*/5)(b^2+c^2),  I_yy = (M*/5)(a^2+c^2),  I_zz = (M*/5)(a^2+b^2)
# with semi-axes (a,b,c) along (x,y,z).  Then  M*/Izz = 5/(a^2+b^2),  and the
# estimated inertia ratios fix the shape:
#     Ixx/Izz + Iyy/Izz = 1 + 2 c^2/(a^2+b^2)
#   =>  a^2+b^2 = 2 c^2 / (Ixx/Izz + Iyy/Izz - 1)
#   =>  M*/Izz = 5 (Ixx/Izz + Iyy/Izz - 1) / (2 c^2)
#             = 10 (Ixx/Izz + Iyy/Izz - 1) / L^2 ,
# where z is the long (minimum-inertia) spin axis and L = 2c is Toutatis'
# long-axis length (4.6 km, Hudson et al. 2003 shape model; consistent with
# the 4 km scale bar in Fig. 1).  This ties the only free scale to published
# numbers instead of an arbitrary guess.
_IXX_R, _IYY_R = 3.091, 3.2178             # a-priori inertia ratios (Table 2)
_LONG_AXIS_KM = 4.6                        # Toutatis long-axis length
MSTAR_OVER_IZZ = 10.0 * (_IXX_R + _IYY_R - 1.0) / _LONG_AXIS_KM**2   # 1/km^2

# ---------------------------------------------------------------------------
# Estimation epoch
# ---------------------------------------------------------------------------
EPOCH_UTC = "1992-11-09 17:49:47"
# Julian date (UTC) of the epoch:
from datetime import datetime, timezone
def _to_jd(dt):
    # Julian Date from a UTC datetime (Gregorian).
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    frac = (dt.hour - 12) / 24 + dt.minute / 1440 + dt.second / 86400
    return jdn + frac

EPOCH_JD = _to_jd(datetime(1992, 11, 9, 17, 49, 47))

# ---------------------------------------------------------------------------
# Table 2 : state vector  X = [alpha,beta,gamma, w1,w2,w3,
#                              Ixx,Iyy,Izz,Ixy,Iyz,Ixz, rx,ry,rz]
# The filter estimates everything except Izz (held at 1, sigma=1e-9).
# ---------------------------------------------------------------------------
# name        initial      a-priori 1-sigma     "published" estimate
TABLE2 = [
    ("alpha", 144.863,     15.0,                145.498),     # deg
    ("beta",   65.467,     15.0,                 65.865),     # deg
    ("gamma", 241.785,     15.0,                241.524),     # deg
    ("w1",     14.514,      0.1,                 14.510),     # deg/day
    ("w2",     33.532,      0.1,                 33.529),     # deg/day
    ("w3",    -98.713,      0.1,                -98.709),     # deg/day
    ("Ixx",     3.091,      1e-1,                 3.0836),    # -
    ("Iyy",     3.2178,     1e-1,                 3.235),     # -
    ("Izz",     1.0,        1e-9,                 1.0),       # - (held)
    ("Ixy",     0.0,        1e-2,                -7.1082e-4), # -
    ("Iyz",     0.0,        1e-2,                 1.1707e-3), # -
    ("Ixz",     0.0,        1e-2,                 1.3252e-3), # -
    ("rx",      0.0,        1e-3,                 5.126e-7),  # km
    ("ry",      0.0,        1e-3,                -1.720e-7),  # km
    ("rz",      0.0,        1e-3,                -3.6732e-7), # km
]
STATE_NAMES = [r[0] for r in TABLE2]
X0_INITIAL  = np.array([r[1] for r in TABLE2], float)
SIGMA_APRIORI = np.array([r[2] for r in TABLE2], float)
X_PUBLISHED = np.array([r[3] for r in TABLE2], float)

# Index of Izz (held fixed - excluded from the free parameter set)
IZZ_IDX = STATE_NAMES.index("Izz")
FREE_IDX = [i for i in range(len(STATE_NAMES)) if i != IZZ_IDX]   # 14 free params

# ---------------------------------------------------------------------------
# Table 3 : observation log.  Each entry: (year, month, day, "HH:MM",
#           (alpha,beta,gamma) deg, (w1,w2,w3) deg/day, station)
# Angular velocities are recorded for validation only - they are NOT fit.
# ---------------------------------------------------------------------------
_MON = dict(Jan=1, Feb=2, Mar=3, Apr=4, May=5, Jun=6,
            Jul=7, Aug=8, Sep=9, Oct=10, Nov=11, Dec=12)

TABLE3 = [
    # 1992 Goldstone
    (1992, "Dec",  2, "21:40", (122.2, 86.5, 107.0), (-35.6,   7.2, -97.0), "Goldstone"),
    (1992, "Dec",  3, "19:30", ( 86.3, 81.8,  24.5), (-16.4, -29.1, -91.9), "Goldstone"),
    (1992, "Dec",  4, "18:10", ( 47.8, 60.7, 284.0), ( 29.1, -23.2, -97.8), "Goldstone"),
    (1992, "Dec",  5, "18:50", ( 14.6, 39.4, 207.1), ( 33.3,   8.2, -92.2), "Goldstone"),
    (1992, "Dec",  6, "17:30", (331.3, 23.7, 151.6), (  6.6,  34.5, -95.8), "Goldstone"),
    (1992, "Dec",  7, "17:20", (222.5, 25.4, 143.9), ( 12.8,  25.4,-104.1), "Goldstone"),
    (1992, "Dec",  8, "16:40", (169.8, 45.5, 106.9), (-31.1, -21.9, -97.7), "Goldstone"),
    (1992, "Dec",  9, "17:50", (137.3, 71.3,  22.3), ( 11.8, -36.9, -94.9), "Goldstone"),
    (1992, "Dec", 10, "17:20", (103.1, 85.2, 292.6), ( 35.8,  -8.9, -97.9), "Goldstone"),
    (1992, "Dec", 11, "09:40", ( 77.0, 85.7, 225.5), ( 31.0,  17.0, -96.3), "Goldstone"),
    (1992, "Dec", 12, "09:20", ( 42.8, 70.2, 133.2), ( -1.3,  37.0, -95.0), "Goldstone"),
    (1992, "Dec", 13, "08:10", ( 13.7, 44.4,  51.9), (-38.3,  17.9, -97.3), "Goldstone"),
    (1992, "Dec", 14, "07:50", (323.7, 14.0,   0.0), (-70.5, -30.6, -91.1), "Goldstone"),
    (1992, "Dec", 15, "07:50", (193.2, 24.4,  21.4), ( 22.1, -26.6, -96.6), "Goldstone"),
    (1992, "Dec", 16, "07:10", (165.1, 46.4, 310.6), ( 33.4,  -3.4, -93.7), "Goldstone"),
    (1992, "Dec", 17, "06:49", (130.6, 76.1, 234.9), ( 12.6,  33.9, -94.0), "Goldstone"),
    (1992, "Dec", 18, "07:09", ( 91.6, 81.6, 142.4), (-24.3,  29.6,-102.0), "Goldstone"),
    # 1996 Goldstone
    (1996, "Nov", 25, "19:48", (130.5, 78.9, 143.2), (-32.0,  16.4, -98.2), "Goldstone"),
    (1996, "Nov", 26, "17:51", ( 94.2, 88.1,  57.7), (-30.6, -18.7, -91.9), "Goldstone"),
    (1996, "Nov", 27, "17:34", ( 60.4, 81.2, 320.9), ( 10.7, -36.8, -94.7), "Goldstone"),
    (1996, "Nov", 29, "15:37", (349.3, 30.0, 168.0), ( 23.1,  28.9, -98.3), "Goldstone"),
    (1996, "Nov", 30, "15:47", (250.3, 14.2, 166.9), (-18.6,  32.1, -94.9), "Goldstone"),
    (1996, "Dec",  1, "14:23", (180.4, 37.6, 139.3), (-38.7,  -0.5, -98.1), "Goldstone"),
    (1996, "Dec",  2, "13:43", (146.7, 64.0,  64.9), (-12.6, -34.8, -97.9), "Goldstone"),
    (1996, "Dec",  3, "12:20", (116.7, 81.4, 340.4), ( 24.3, -28.2, -98.1), "Goldstone"),
    # 2000 Goldstone
    (2000, "Nov",  4, "17:06", (110.0, 88.5,  30.0), (  0.0, -32.5, -98.9), "Goldstone"),
    (2000, "Nov",  5, "18:01", ( 70.6, 84.0, 281.0), ( 34.5, -17.2, -97.9), "Goldstone"),
    # 2004 Arecibo
    (2004, "Oct",  7, "13:56", ( 79.9, 85.3, 365.2), ( -2.5, -35.4,-109.0), "Arecibo"),
    (2004, "Oct",  8, "14:04", ( 44.9, 72.5, 263.1), ( 32.4, -18.1, -97.9), "Arecibo"),
    (2004, "Oct",  9, "13:57", ( 12.8, 47.3, 181.4), ( 29.7,  -2.0, -98.1), "Arecibo"),
    (2004, "Oct", 10, "13:17", (327.7, 20.4, 124.1), (-10.7,  34.7, -97.3), "Arecibo"),
    # 2008 Arecibo  (Nov 22 spin vector is anomalous in the table; w is unused)
    (2008, "Nov", 22, "10:54", (119.5, 90.7,  92.0), (118.1,  90.4,  93.6), "Arecibo"),
    (2008, "Nov", 23, "10:45", ( 86.2, 85.0,   0.3), ( -0.4, -36.2, -98.9), "Arecibo"),
]


def observation_arrays():
    """Return (jd, euler_deg[N,3], omega_deg_day[N,3], sigma_deg[N])."""
    jd, eul, om, sig = [], [], [], []
    for (yr, mon, day, hhmm, e, w, station) in TABLE3:
        h, mi = map(int, hhmm.split(":"))
        jd.append(_to_jd(datetime(yr, _MON[mon], day, h, mi, 0)))
        eul.append(e)
        om.append(w)
        # observation uncertainty inflation (Section 6):
        # 15 deg for 1992-2000, 20 deg for 2004-2008
        sig.append(15.0 if yr <= 2000 else 20.0)
    return (np.array(jd), np.array(eul, float),
            np.array(om, float), np.array(sig, float))
