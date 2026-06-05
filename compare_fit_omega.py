"""
Demonstration: what happens when the angular velocity is ALSO put into the
fit, instead of being held back for validation (the paper's choice).

Runs the batch filter twice from the same a-priori and prints the formal 1-sigma
uncertainties side by side.  Because the radar spin vector is derived from the
same orientation imagery as the Euler angles, adding it double-counts
information and artificially shrinks the covariance - watch the sigmas drop.

(The covariance is set by the observation geometry at the first linearization,
so n_iter=1 already gives representative sigmas; this keeps the demo short.)
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dynamics import Model, Ephemeris
import filter as F
import data_paper as D

DEG = np.pi / 180.0


def run(fit_omega):
    eph = Ephemeris(os.path.join(HERE, "data/ephem_sun.npz"),
                    os.path.join(HERE, "data/ephem_earth.npz"))
    model = Model(eph)
    jd, eul, om, sig = D.observation_arrays()
    res = F.run_filter(model, jd, eul, sig, n_iter=1, verbose=False,
                       obs_weight=1.0, fit_omega=fit_omega, obs_omega_deg=om)
    sd = np.sqrt(np.diag(res["P"]))
    # to paper units
    out = np.zeros(len(D.STATE_NAMES))
    for k, idx in enumerate(D.FREE_IDX):
        s = sd[k]
        if D.STATE_NAMES[idx] in ("alpha", "beta", "gamma", "w1", "w2", "w3"):
            s /= DEG
        out[idx] = s
    return out


def main():
    print("Running angles-only fit ...", flush=True)
    s_ang = run(False)
    print("Running angles + omega fit ...", flush=True)
    s_aw = run(True)

    print("\n" + "=" * 70)
    print("Formal 1-sigma uncertainty:  angles-only  vs  angles+omega")
    print("=" * 70)
    print(f"{'param':6} {'angles-only':>14} {'angles+omega':>14} {'ratio':>8}"
          f" {'published':>11}")
    pub = ["3.762", "2.388", "2.586", "0.0994", "0.0971", "0.0957",
           "0.02822", "0.0714", "1e-9", "0.00994", "0.00939", "0.00753",
           "1.68e-5", "3.35e-5", "4.29e-5"]
    for i, nm in enumerate(D.STATE_NAMES):
        r = s_aw[i] / s_ang[i] if s_ang[i] > 0 else float("nan")
        print(f"{nm:6} {s_ang[i]:14.4g} {s_aw[i]:14.4g} {r:8.2f} {pub[i]:>11}")
    print("\nratio < 1 => adding omega *shrinks* the formal uncertainty "
          "(over-optimistic,\nbecause omega is not independent of the fitted "
          "Euler-angle data).")


if __name__ == "__main__":
    main()
