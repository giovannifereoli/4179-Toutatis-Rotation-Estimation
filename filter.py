"""
State + STM propagation and the batch least-squares filter
(Takahashi, Busch & Scheeres 2013, Section 5; Eqs. 21-34).

Internal units: angles [rad], angular velocity [rad/day], inertia [-],
offset [km], time [days past EPOCH_JD].

Free (estimated) parameter ordering (14, Izz held at 1):
    [alpha, beta, gamma, w1, w2, w3, Ixx, Iyy, Ixy, Iyz, Ixz, rx, ry, rz]
"""

import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from dynamics import Model, Ephemeris
import data_paper as D

DEG = np.pi / 180.0
NFREE = 14


# unit conversion of an *internal* free-state vector <-> paper (deg) units
def free_to_params9(xf):
    """xf(14) -> params9 = [Ixx,Iyy,Izz,Ixy,Iyz,Ixz, rx,ry,rz] for the RHS."""
    Ixx, Iyy, Ixy, Iyz, Ixz = xf[6:11]
    rx, ry, rz = xf[11:14]
    return np.array([Ixx, Iyy, 1.0, Ixy, Iyz, Ixz, rx, ry, rz])


def x0_internal():
    """A-priori free state (14) in internal units from Table 2."""
    x = D.X0_INITIAL.copy().astype(float)
    # convert angles & rates to rad / rad-per-day
    x[0:3] *= DEG
    x[3:6] *= DEG
    return x[D.FREE_IDX]


def sigma_internal():
    s = D.SIGMA_APRIORI.copy().astype(float)
    s[0:3] *= DEG
    s[3:6] *= DEG
    return s[D.FREE_IDX]


# ===========================================================================
#  Propagation of  s(6) + STM Phi(14x14)
# ===========================================================================
def propagate(model, xf, t_eval, rtol=1e-10, atol=1e-12, max_step=0.5,
              progress=False):
    """Integrate the dynamic 6-state and the 14x14 STM from epoch (t=0) to
    each time in t_eval (sorted, >0).  Returns:
        S   : (len(t_eval), 6)  dynamic state at each time
        PHI : (len(t_eval), 14, 14) STM Phi(t_k, t0)
    """
    params = free_to_params9(xf)
    s0 = xf[0:6].copy()
    Phi0 = np.eye(NFREE)
    z0 = np.concatenate([s0, Phi0.ravel()])

    T = float(t_eval[-1])
    _pg = {"last": time.time(), "tmax": 0.0}

    def rhs(t, z):
        s = z[0:6]
        Phi = z[6:].reshape(NFREE, NFREE)
        sdot = model.fdot(t, s, params)
        A = model.Amat(t, s, params)
        Phidot = A @ Phi
        if progress:
            _pg["tmax"] = max(_pg["tmax"], t)
            now = time.time()
            if now - _pg["last"] > 3.0:
                print(
                    f"\r    integrating: t={_pg['tmax']:7.0f}/{T:.0f} d "
                    f"({100*_pg['tmax']/T:5.1f}%)",
                    end="",
                    flush=True,
                )
                _pg["last"] = now
        return np.concatenate([sdot, Phidot.ravel()])

    sol = solve_ivp(
        rhs,
        (0.0, t_eval[-1]),
        z0,
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    Z = sol.y.T
    S = Z[:, 0:6]
    PHI = Z[:, 6:].reshape(-1, NFREE, NFREE)
    return S, PHI


# ===========================================================================
#  Batch least-squares filter  (Gauss-Newton on Eq. 34)
# ===========================================================================
def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_filter(model, obs_jd, obs_eul_deg, obs_sig_deg, n_iter=8, verbose=True,
               lm=0.0, obs_weight=1.0, prior_weight=1.0,
               fit_omega=False, obs_omega_deg=None, sigma_omega_deg=None):
    """Fit the 3-1-3 Euler-angle observations.  Returns dict with the
    converged free state, covariance, and per-iteration diagnostics.

    obs_weight   : >1 tightens the observation uncertainties (sigma_eff =
                   sigma_obs / obs_weight), i.e. trusts the data more relative
                   to the a-priori.  This also rescales the *normalized*
                   residual plots by obs_weight, so the post-fit points spread
                   out toward the +/-3 sigma band instead of sitting < 1 sigma.
    prior_weight : >1 tightens the a-priori covariance (P0_eff = P0/prior_weight),
                   i.e. holds the solution closer to the Table-2 initial values.
    Note: the paper inflated the observation sigmas to 15/20 deg precisely to
    make the long-arc batch converge, so obs_weight beyond a few may reintroduce
    convergence trouble - raise it gradually.

    fit_omega    : EXPERIMENTAL (default False = the paper's angles-only fit).
                   If True, the body angular velocity is ALSO added to the
                   observation vector: the partial rows H~_w = Phi[3:6,:] are
                   stacked under the Euler-angle rows.  This is *not* the paper's
                   approach - the radar spin vector is derived from the same
                   orientation imagery as the angles (so it double-counts
                   information and over-tightens the covariance), and its true
                   per-observation uncertainties are geometry dependent and were
                   omitted from Table 3.  Provide obs_omega_deg (N,3).
    sigma_omega_deg : per-component omega sigma [deg/day].  Scalar, (N,) or
                   (N,3).  If None, a geometry-aware default is used: the
                   transverse components (w1,w2) get sigma = clip(2/sin^2(beta),
                   2, 30) deg/day (they are ill-determined near the end-on /
                   small-beta geometry), while w3 keeps 2 deg/day.  This stands
                   in for Table 3's omitted uncertainties; it is an approximation."""
    t_obs = obs_jd - D.EPOCH_JD  # days past epoch
    order = np.argsort(t_obs)
    t_obs = t_obs[order]
    y_obs = obs_eul_deg[order] * DEG  # rad, shape (N,3)
    sig = obs_sig_deg[order] * DEG / obs_weight   # effective (weighted) sigma
    Winv = sig**2  # per-obs variance

    # ---- optional angular-velocity observations ----
    if fit_omega:
        if obs_omega_deg is None:
            raise ValueError("fit_omega=True requires obs_omega_deg")
        w_obs = obs_omega_deg[order] * DEG          # rad/day, (N,3)
        beta = obs_eul_deg[order, 1]                # deg
        if sigma_omega_deg is None:
            sperp = np.clip(2.0 / np.sin(np.deg2rad(beta)) ** 2, 2.0, 30.0)
            sigw = np.column_stack([sperp, sperp, np.full(len(beta), 2.0)])
        else:
            sigw = np.broadcast_to(np.asarray(sigma_omega_deg, float),
                                   (len(beta), 3)).copy()
        sigw = sigw * DEG / obs_weight              # rad/day, (N,3)
        Ww = 1.0 / sigw ** 2                         # per-component weight

    xbar = x0_internal()  # a-priori mean (free)
    P0 = np.diag(sigma_internal() ** 2)
    P0inv = np.linalg.inv(P0) * prior_weight
    xref = xbar.copy()

    history = []
    arc = t_obs[-1]
    for it in range(n_iter):
        if verbose:
            print(
                f"  iter {it}: propagating {len(t_obs)} obs over " f"{arc:.0f} d ...",
                flush=True,
            )
        t0 = time.time()
        S, PHI = propagate(model, xref, t_obs, progress=verbose)
        if verbose:
            print(
                f"\r    propagation done in {time.time()-t0:5.1f}s" + " " * 20,
                flush=True,
            )
        comp = S[:, 0:3]  # computed Euler angles
        # residuals (observed - computed), wrapped
        y = wrap_pi(y_obs - comp)  # (N,3)
        if it == 0:  # pre-fit (a-priori state)
            prefit_norm = y / sig[:, None]
            prefit_w = S[:, 3:6] / DEG  # deg/day

        # angular-velocity residuals (no wrapping - these are rates)
        yw = (w_obs - S[:, 3:6]) if fit_omega else None

        Lam = P0inv.copy()
        N = P0inv @ (xbar - xref)
        rms_norm = []
        for k in range(len(t_obs)):
            Hk = PHI[k][0:3, :]  # H~ Phi = Phi[0:3,:]
            w = 1.0 / Winv[k]
            Lam += w * (Hk.T @ Hk)
            N += w * (Hk.T @ y[k])
            rms_norm.append(y[k] / sig[k])
            if fit_omega:                      # stack the omega rows H~_w = Phi[3:6,:]
                Hw = PHI[k][3:6, :]
                Lam += (Hw.T * Ww[k]) @ Hw     # Hw^T diag(Ww) Hw
                N += Hw.T @ (Ww[k] * yw[k])
        # optional Levenberg-Marquardt damping for robustness on the long,
        # highly phase-sensitive arc
        Lam_s = Lam + lm * np.diag(np.diag(Lam))
        dx = np.linalg.solve(Lam_s, N)
        P = np.linalg.inv(Lam)
        xref = xref + dx

        rms = np.sqrt(np.mean(np.array(rms_norm) ** 2))
        step = np.linalg.norm(dx / sigma_internal())
        history.append(
            dict(it=it, rms_norm=rms, step=step, resid_std=np.std(np.array(rms_norm)))
        )
        if verbose:
            print(f"    normalized-RMS={rms:7.4f}  |dx/sigma|={step:.3e}", flush=True)
        if step < 1e-4:
            break

    # final residuals at converged solution
    if verbose:
        print("  computing post-fit residuals ...", flush=True)
    S, PHI = propagate(model, xref, t_obs, progress=verbose)
    y_final = wrap_pi(y_obs - S[:, 0:3])
    resid_norm = y_final / sig[:, None]
    return dict(
        xref=xref,
        P=P,
        history=history,
        t_obs=t_obs,
        resid_norm=resid_norm,
        S=S,
        comp_eul=S[:, 0:3],
        comp_w=S[:, 3:6],
        prefit_norm=prefit_norm,
        prefit_w=prefit_w,
        fit_omega=fit_omega,
        omega_sigma_deg=(sigw / DEG) if fit_omega else None,
    )


# ===========================================================================
#  Finite-difference validation of the dynamics matrix A
# ===========================================================================
def fd_check_A(model, t, s, params, eps=1e-7):
    """Compare analytic Amat against a central finite-difference Jacobian of
    the augmented RHS [fdot ; 0] w.r.t. the 14 free variables."""

    def aug_rhs(z14):
        s_ = z14[0:6]
        p9 = free_to_params9(z14)
        return model.fdot(t, s_, p9)  # 6-vector

    z = np.concatenate([s, params[[0, 1, 3, 4, 5]], params[6:9]])  # 14 free
    Aana = model.Amat(t, s, params)
    Jnum = np.zeros((6, 14))
    for j in range(14):
        zp = z.copy()
        zm = z.copy()
        h = eps * max(1.0, abs(z[j]))
        zp[j] += h
        zm[j] -= h
        Jnum[:, j] = (aug_rhs(zp) - aug_rhs(zm)) / (2 * h)
    return Aana[0:6, :], Jnum
