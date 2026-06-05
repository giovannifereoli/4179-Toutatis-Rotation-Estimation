"""
Rotational dynamics of a tumbling rigid body subject to solar + terrestrial
tidal torques, following Takahashi, Busch & Scheeres (2013), Sections 2 & 5
and Appendix A.

State (estimated) vector, 15 components (Eq. 23), Izz held fixed:
    X = [alpha, beta, gamma,      3-1-3 Euler angles            [rad]
         w1, w2, w3,              body-frame angular velocity   [rad/day]
         Ixx, Iyy, Izz,           inertia ratios (Izz == 1)     [-]
         Ixy, Iyz, Ixz,
         rx, ry, rz]              COM-COF offset                [km]

Internally the propagated dynamic state is s = [alpha,beta,gamma, w1,w2,w3];
the inertia ratios and offset are constant parameters that enter the RHS.

All angles in radians, time in days, distance in km internally.
"""
import numpy as np
from data_paper import (GM_SUN, GM_EARTH, MSTAR_OVER_IZZ, EPOCH_JD)

DEG = np.pi / 180.0


# ===========================================================================
#  Ephemeris interpolation
# ===========================================================================
class Ephemeris:
    """Holds the Horizons samples and returns the inertial vector r FROM
    Toutatis TO the perturbing body (km) at an arbitrary time t [days past
    EPOCH_JD].

    Sun  : 1-day samples, two-body f-g series interpolation (Section 3).
    Earth: 30-min samples, linear interpolation; the terrestrial torque is
           only active within +/-1 month of an Earth flyby, so outside the
           sampled windows the Earth contribution is reported as inactive.
    """
    def __init__(self, sun_npz, earth_npz):
        s = np.load(sun_npz)
        self.sun_t = s["jd"] - EPOCH_JD            # days past epoch
        self.sun_r = s["pos"]                      # Toutatis wrt Sun  [km]
        self.sun_v = s["vel"]                      # [km/day]
        e = np.load(earth_npz)
        self.ear_t = e["jd"] - EPOCH_JD
        self.ear_r = e["pos"]                      # Toutatis wrt Earth [km]
        # window edges of the (sorted) earth samples, used to mask the torque
        self.ear_tmin, self.ear_tmax = self.ear_t[0], self.ear_t[-1]

    # ---- Sun : f-g two-body interpolation from the nearest 1-day node ----
    def sun_vec(self, t):
        i = int(np.clip(np.searchsorted(self.sun_t, t) - 1, 0,
                        len(self.sun_t) - 2))
        # pick the closer of node i / i+1 as the propagation anchor
        if abs(t - self.sun_t[i + 1]) < abs(t - self.sun_t[i]):
            i += 1
        r0 = self.sun_r[i]; v0 = self.sun_v[i]; dt = t - self.sun_t[i]
        r = _fg_propagate(r0, v0, dt, GM_SUN)
        return -r                                  # Toutatis -> Sun

    # ---- Earth : linear interpolation; None outside the sampled windows --
    def earth_vec(self, t):
        if t < self.ear_tmin or t > self.ear_tmax:
            return None
        j = int(np.clip(np.searchsorted(self.ear_t, t) - 1, 0,
                        len(self.ear_t) - 2))
        t0, t1 = self.ear_t[j], self.ear_t[j + 1]
        if t1 - t0 > 0.1:        # gap between flyby windows -> torque off
            return None
        f = (t - t0) / (t1 - t0)
        r = (1 - f) * self.ear_r[j] + f * self.ear_r[j + 1]
        return -r                                  # Toutatis -> Earth


def _fg_propagate(r0, v0, dt, mu):
    """Two-body f-g series propagation of position over a short interval dt
    (Danby 1962, ch. 6).  Universal-variable Kepler solve."""
    if dt == 0.0:
        return r0.copy()
    r0n = np.linalg.norm(r0)
    v0s = v0 @ v0
    alpha = 2.0 / r0n - v0s / mu          # 1/a
    sqmu = np.sqrt(mu)
    sigma0 = (r0 @ v0) / sqmu
    # initial guess for universal anomaly chi
    chi = sqmu * abs(alpha) * dt
    for _ in range(60):
        psi = chi * chi * alpha
        c2, c3 = _stumpff(psi)
        r = chi * chi * c2 + sigma0 * chi * (1 - psi * c3) + r0n * (1 - psi * c2)
        dchi = (sqmu * dt - chi**3 * c3 - sigma0 * chi**2 * c2
                - r0n * chi * (1 - psi * c3)) / r
        chi += dchi
        if abs(dchi) < 1e-12:
            break
    f = 1.0 - chi * chi / r0n * c2
    g = dt - chi**3 / sqmu * c3
    return f * r0 + g * v0


def _stumpff(psi):
    if psi > 1e-6:
        s = np.sqrt(psi)
        c2 = (1 - np.cos(s)) / psi
        c3 = (s - np.sin(s)) / s**3
    elif psi < -1e-6:
        s = np.sqrt(-psi)
        c2 = (1 - np.cosh(s)) / psi
        c3 = (np.sinh(s) - s) / s**3
    else:
        c2 = 0.5; c3 = 1.0 / 6.0
    return c2, c3


# ===========================================================================
#  Kinematics / rotation matrices
# ===========================================================================
def tilde(a):
    """Cross-product (skew) matrix [a~] (Eq. 7)."""
    return np.array([[0.0, -a[2], a[1]],
                     [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]])


def R1(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def R3(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def BN(alpha, beta, gamma):
    """Inertial -> body rotation matrix, 3-1-3 Euler angles (Eq. 1)."""
    return R3(gamma) @ R1(beta) @ R3(alpha)


def Cmat(alpha, beta, gamma):
    """[C(alpha)] such that alpha_dot = [C] w_B  (Eq. 4)."""
    sb = np.sin(beta)
    sg, cg = np.sin(gamma), np.cos(gamma)
    M = np.array([[sg,          cg,          0.0],
                  [cg * sb,    -sg * sb,     0.0],
                  [-sg * np.cos(beta), -cg * np.cos(beta), sb]])
    return M / sb


# ===========================================================================
#  Inertia tensor from the estimated ratios
# ===========================================================================
def inertia_tensor(p):
    """p = [Ixx,Iyy,Izz,Ixy,Iyz,Ixz] -> symmetric 3x3."""
    Ixx, Iyy, Izz, Ixy, Iyz, Ixz = p
    return np.array([[Ixx, Ixy, Ixz],
                     [Ixy, Iyy, Iyz],
                     [Ixz, Iyz, Izz]])


# basis derivatives dI/dI_ij for the 5 estimated inertia ratios
def _dI(name):
    E = np.zeros((3, 3))
    idx = dict(Ixx=(0, 0), Iyy=(1, 1), Ixy=(0, 1), Iyz=(1, 2), Ixz=(0, 2))
    i, j = idx[name]
    E[i, j] = 1.0
    if i != j:
        E[j, i] = 1.0
    return E
DINERTIA = {n: _dI(n) for n in ("Ixx", "Iyy", "Ixy", "Iyz", "Ixz")}


# ===========================================================================
#  Torques  (Eqs. 17 & 20)
# ===========================================================================
def torque_body(alpha, beta, gamma, I, r_cm, r_in, GM):
    """Tidal torque per Izz from one external body, in the BODY frame.

    L_hat = L / Izz, summing the first-degree (COM-COF, Eq.17) and
    second-degree (inertia, Eq.20) contributions.

    r_in : inertial vector Toutatis -> perturber [km]; GM [km^3/day^2].
    Returns L_hat [rad/day^2-like, body frame].
    """
    rb = BN(alpha, beta, gamma) @ r_in        # body-frame position of perturber
    rn = np.linalg.norm(rb)
    rt = tilde(rb)
    # second degree (Eq. 20):  L2 = 3 G Ms / r^5 [r~][I] r        (divided by Izz)
    L2 = (3.0 * GM / rn**5) * (rt @ (I @ rb))
    # first degree (Eq. 17):   L1 = -G M* Ms / r^3 [r~] r_cm      (divided by Izz)
    L1 = -(GM * MSTAR_OVER_IZZ / rn**3) * (rt @ r_cm)
    return L1 + L2, rb, rn, rt


# ===========================================================================
#  Full RHS  s_dot = F(s; params, t)   and dynamics matrix A
# ===========================================================================
class Model:
    def __init__(self, ephem):
        self.eph = ephem

    def _perturbers(self, t):
        """List of (GM, r_in) active at time t."""
        out = [(GM_SUN, self.eph.sun_vec(t))]
        re = self.eph.earth_vec(t)
        if re is not None:
            out.append((GM_EARTH, re))
        return out

    # -------- state derivative (dynamic 6-state) ----------------------
    def fdot(self, t, s, params):
        alpha, beta, gamma, w1, w2, w3 = s
        w = np.array([w1, w2, w3])
        I = inertia_tensor(params[:6])
        r_cm = params[6:9]
        # kinematics
        adot = Cmat(alpha, beta, gamma) @ w
        # torque (sum of perturbers)
        L = np.zeros(3)
        for GM, r_in in self._perturbers(t):
            Lk, *_ = torque_body(alpha, beta, gamma, I, r_cm, r_in, GM)
            L += Lk
        # Euler's equation (Eq. 8) with I -> Ixx..Izz (Izz cancels overall)
        Iinv = np.linalg.inv(I)
        wdot = Iinv @ (-tilde(w) @ (I @ w) + L)
        return np.concatenate([adot, wdot])

    # -------- dynamics matrix A = dFdot/dZ  (Appendix A) --------------
    # Augmented integration state Z = [s(6), Ixx,Iyy,Ixy,Iyz,Ixz(5), rx,ry,rz(3)]
    # i.e. 14 components matching the free parameter set (Izz held).
    def Amat(self, t, s, params):
        alpha, beta, gamma, w1, w2, w3 = s
        w = np.array([w1, w2, w3])
        I = inertia_tensor(params[:6])
        Iinv = np.linalg.inv(I)
        r_cm = params[6:9]
        sb, cb = np.sin(beta), np.cos(beta)
        sg, cg = np.sin(gamma), np.cos(gamma)

        A = np.zeros((14, 14))
        # --- d(alpha_dot)/d(alpha,beta,gamma)  (A1-A3) -----------------
        # d/d alpha = 0  (A1)
        # d/d beta  (A2)
        dadot_db = (1.0 / sb**2) * np.array([[-sg * cb, -cg * cb, 0.0],
                                             [0.0,       0.0,     0.0],
                                             [sg,        cg,      0.0]]) @ w
        # d/d gamma (A3)
        dadot_dg = np.array([[cg / sb,  -sg / sb,  0.0],
                             [-sg,      -cg,       0.0],
                             [-cg * cb / sb, sg * cb / sb, 0.0]]) @ w
        A[0:3, 1] = dadot_db
        A[0:3, 2] = dadot_dg
        # --- d(alpha_dot)/d(omega) = [C(alpha)]  (A4) ------------------
        A[0:3, 3:6] = Cmat(alpha, beta, gamma)
        # --- d(omega_dot)/d(omega)  (A6) -------------------------------
        Iw = I @ w
        A[3:6, 3:6] = Iinv @ (tilde(Iw) - tilde(w) @ I)
        # --- d(omega_dot)/d(inertia ratios)  (A7/A10 direct form) ------
        # wdot = Iinv (-[w~] I w + L) ;  dL/dIij from 2nd-degree term only
        rhs = -tilde(w) @ Iw + self._L(t, alpha, beta, gamma, I, r_cm)
        for col, name in zip(range(6, 11), ("Ixx", "Iyy", "Ixy", "Iyz", "Ixz")):
            dIm = DINERTIA[name]
            dL = self._dL_dI(t, alpha, beta, gamma, dIm)
            term = (-Iinv @ dIm @ Iinv @ rhs
                    + Iinv @ (-tilde(w) @ (dIm @ w) + dL))
            A[3:6, col] = term
        # --- d(omega_dot)/d(r_cm)  (A21,A22) ---------------------------
        dwdot_dr = np.zeros((3, 3))
        for GM, r_in in self._perturbers(t):
            rb = BN(alpha, beta, gamma) @ r_in
            rn = np.linalg.norm(rb)
            # dL1/dr_cm = -(G M* Ms / r^3)[r~]   (A22, per Izz)
            dwdot_dr += Iinv @ (-(GM * MSTAR_OVER_IZZ / rn**3) * tilde(rb))
        A[3:6, 11:14] = dwdot_dr
        return A

    # helper: total torque (per Izz) at a state
    def _L(self, t, alpha, beta, gamma, I, r_cm):
        L = np.zeros(3)
        for GM, r_in in self._perturbers(t):
            Lk, *_ = torque_body(alpha, beta, gamma, I, r_cm, r_in, GM)
            L += Lk
        return L

    # helper: d(total 2nd-degree torque)/dI_ij = sum 3GM/r^5 [r~] dI r
    def _dL_dI(self, t, alpha, beta, gamma, dIm):
        dL = np.zeros(3)
        for GM, r_in in self._perturbers(t):
            rb = BN(alpha, beta, gamma) @ r_in
            rn = np.linalg.norm(rb)
            dL += (3.0 * GM / rn**5) * (tilde(rb) @ (dIm @ rb))
        return dL
