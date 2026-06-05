"""
Download JPL Horizons ephemerides for 4179 Toutatis, as used by
Takahashi, Busch & Scheeres (2013), AJ 146:95.

The paper (Section 3) retrieves:
  * Toutatis position relative to EARTH in 30-minute increments
    (used for the terrestrial tidal torque, linearly interpolated),
  * Toutatis position relative to the SUN in 1-day increments
    (used for the solar tidal torque, f-g series interpolated).

We pull state VECTORS (position + velocity) in the Earth mean-equator /
equinox of J2000 frame (refplane='earth'), which is the inertial frame in
which the 3-1-3 Euler angles are defined.

Vectors returned by Horizons are TARGET relative to CENTER (Toutatis wrt
center).  In the dynamics we need the vector from Toutatis TO the perturbing
body, r = -(Toutatis wrt body); the sign flip is applied at use time.

Output: data/ephem_sun.npz, data/ephem_earth.npz  (km, km/day).
"""
import numpy as np
from astroquery.jplhorizons import Horizons
import os

AU_KM = 149597870.7
DAY_S = 86400.0
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ----------------------------------------------------------------------
# Earth-relative windows: terrestrial torque is only active ~+/-1 month
# around each Earth flyby (1992, 1996, 2000, 2004, 2008).  30-min steps.
# ----------------------------------------------------------------------
EARTH_WINDOWS = [
    ("1992-10-01", "1992-12-31"),
    ("1996-10-15", "1996-12-10"),
    ("2000-09-20", "2000-11-15"),
    ("2004-08-20", "2004-10-25"),
    ("2008-10-15", "2008-12-01"),
]
SUN_SPAN = ("1992-10-01", "2009-01-01")  # whole data arc, 1-day steps


def _fetch(center, start, stop, step):
    obj = Horizons(id="4179", id_type="smallbody", location=center,
                   epochs={"start": start, "stop": stop, "step": step})
    v = obj.vectors(refplane="earth")
    col = lambda c: np.asarray(np.ma.getdata(v[c]), float)
    jd = col("datetime_jd")
    pos = np.column_stack([col("x"), col("y"), col("z")]) * AU_KM       # km
    vel = np.column_stack([col("vx"), col("vy"), col("vz")]) * AU_KM    # km/day
    return jd, pos, vel


def main():
    # --- Sun (heliocentric), 1-day ---
    print("Downloading heliocentric ephemeris (1-day)...")
    jd, pos, vel = _fetch("@sun", *SUN_SPAN, "1d")
    np.savez(os.path.join(DATA, "ephem_sun.npz"), jd=jd, pos=pos, vel=vel)
    print(f"  Sun: {len(jd)} samples, {jd[0]:.1f} .. {jd[-1]:.1f}")

    # --- Earth (geocentric), 30-min, per window ---
    print("Downloading geocentric ephemeris (30-min) per flyby window...")
    jds, poss, vels = [], [], []
    for (a, b) in EARTH_WINDOWS:
        jd, pos, vel = _fetch("@399", a, b, "30m")
        jds.append(jd); poss.append(pos); vels.append(vel)
        print(f"  Earth {a}..{b}: {len(jd)} samples")
    jd = np.concatenate(jds); pos = np.vstack(poss); vel = np.vstack(vels)
    order = np.argsort(jd)
    np.savez(os.path.join(DATA, "ephem_earth.npz"),
             jd=jd[order], pos=pos[order], vel=vel[order])
    print(f"  Earth total: {len(jd)} samples")
    print("Done.")


if __name__ == "__main__":
    main()
