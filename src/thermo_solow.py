"""Reference implementation of the thermo-ecological Solow model.

Implements the core model of:

    Greenlee, S. (2026). "Thermodynamic Waste, Health Capital, and Long-Run
    Growth: A Thermo-Ecological Augmentation of the Solow Model."

Equation numbers in comments and docstrings refer to the paper. The module
covers the analytic core (Eqs. 9, 11, 13-19), the transition dynamics of
Appendix B.3, and the health-capital fixed point of Appendix B.4. Run
verify.py to check every proposition numerically.

Only dependency: numpy.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Params:
    """Model parameters.

    Solow block (standard): alpha, s, n, g, delta.
    Thermo-ecological block: tau (TO intensity, Eq. 3), psi (penalty
    curvature, Eq. 9, axiom A3 requires psi >= 1), b and m (health
    exponents for the labor-quality and TFP channels, Eq. 7), H_bar
    (baseline health).
    Health law of motion (Eq. 6): eta0, eta1, h, delta_H, chi, kappa.
    """

    alpha: float = 0.30    # capital share
    s: float = 0.21        # savings rate
    n: float = 0.010       # labor force growth
    g: float = 0.018       # Harrod-neutral technical progress
    delta: float = 0.050   # capital depreciation

    tau: float = 0.15      # TO intensity, tau in [0, 1)
    psi: float = 1.5       # phi curvature, psi >= 1 (axiom A3)
    b: float = 0.40        # health -> labor quality exponent
    m: float = 0.20        # health -> TFP exponent
    H_bar: float = 1.0     # baseline health capital

    eta0: float = 0.008    # baseline health improvement
    eta1: float = 0.10     # preventive-spending effectiveness
    h: float = 0.05        # preventive health spending share of output
    delta_H: float = 0.020 # health depreciation
    chi: float = 0.50      # health damage per unit exposure
    kappa: float = 1.0     # exposure dose-response scaling

    def __post_init__(self):
        if not 0.0 <= self.tau < 1.0:
            raise ValueError(f"tau must be in [0, 1), got {self.tau}")
        if self.psi < 1.0:
            raise ValueError(
                f"psi must be >= 1 (axiom A3: phi convex); got {self.psi}"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

    @property
    def break_even(self) -> float:
        """n + g + delta, the effective depreciation of k."""
        return self.n + self.g + self.delta


# ---------------------------------------------------------------------------
# The multiplier and its components (Eqs. 7, 9, 11, 19)
# ---------------------------------------------------------------------------

def phi(tau: float, psi: float) -> float:
    """Sustainability multiplier phi(tau) = (1 - tau)^psi  (Eq. 9).

    Satisfies axioms A1-A4 for psi >= 1: phi(0) = 1, strictly decreasing,
    convex, and phi -> 0 as tau -> 1.
    """
    if psi < 1.0:
        raise ValueError(f"psi must be >= 1 (axiom A3); got {psi}")
    return (1.0 - tau) ** psi


def theta(H: float, H_bar: float, b: float) -> float:
    """Health -> labor quality channel theta(H) = (H / H_bar)^b  (Eq. 7)."""
    return (H / H_bar) ** b


def zeta(H: float, H_bar: float, m: float) -> float:
    """Health -> TFP channel zeta(H) = (H / H_bar)^m  (Eq. 7)."""
    return (H / H_bar) ** m


def omega(p: Params, tau: float = None, H: float = None) -> float:
    """Thermo-ecological multiplier Omega = phi * zeta * theta^(1-alpha)  (Eq. 11).

    theta enters with exponent (1 - alpha) because effective labor A*L*theta
    is raised to (1 - alpha) in the production function.
    """
    tau = p.tau if tau is None else tau
    H = p.H_bar if H is None else H
    return (
        phi(tau, p.psi)
        * zeta(H, p.H_bar, p.m)
        * theta(H, p.H_bar, p.b) ** (1.0 - p.alpha)
    )


def omega_decomposition(p: Params, tau: float = None, H: float = None) -> dict:
    """Decompose Omega into waste, cognitive, and labor drags  (Eq. 19)."""
    tau = p.tau if tau is None else tau
    H = p.H_bar if H is None else H
    return {
        "waste_drag": (1.0 - tau) ** p.psi,
        "cognitive_drag": (H / p.H_bar) ** p.m,
        "labor_drag": (H / p.H_bar) ** (p.b * (1.0 - p.alpha)),
    }


# ---------------------------------------------------------------------------
# Closed-form steady state (Eqs. 15-18) and comparative statics (B.2)
# ---------------------------------------------------------------------------

def k_star(p: Params, Omega: float = None) -> float:
    """Steady-state capital per effective worker  (Eq. 16)."""
    Omega = omega(p) if Omega is None else Omega
    return (p.s * Omega / p.break_even) ** (1.0 / (1.0 - p.alpha))


def y_star(p: Params, Omega: float = None) -> float:
    """Steady-state output per effective worker  (Eq. 18).

    Standard Solow steady state multiplied by Omega^(1/(1-alpha)) - the
    paper's central result: the environmental penalty is amplified because
    it also depresses equilibrium capital accumulation.
    """
    Omega = omega(p) if Omega is None else Omega
    solow_core = (p.s / p.break_even) ** (p.alpha / (1.0 - p.alpha))
    return Omega ** (1.0 / (1.0 - p.alpha)) * solow_core


def amplification_elasticity(p: Params) -> float:
    """d ln(y*) / d ln(Omega) = 1 / (1 - alpha) > 1  (Proposition 4)."""
    return 1.0 / (1.0 - p.alpha)


def dlny_dtau(p: Params) -> float:
    """Analytic total derivative d ln(y*)/d tau = -psi / [(1-alpha)(1-tau)] (Sec. 3.3)."""
    return -p.psi / ((1.0 - p.alpha) * (1.0 - p.tau))


def convergence_rate(p: Params) -> float:
    """lambda_c = (1 - alpha)(n + g + delta), unchanged by the augmentation (B.3)."""
    return (1.0 - p.alpha) * p.break_even


# ---------------------------------------------------------------------------
# Transition dynamics (Eq. 14, Appendix B.3)
# ---------------------------------------------------------------------------

def k_dot(k: float, p: Params, Omega: float = None) -> float:
    """Fundamental equation k_dot = s*Omega*k^alpha - (n+g+delta)*k  (Eq. 14)."""
    Omega = omega(p) if Omega is None else Omega
    return p.s * Omega * k ** p.alpha - p.break_even * k


def simulate_transition(
    p: Params, k0: float, T: float = 400.0, dt: float = 0.01, Omega: float = None
) -> tuple:
    """Integrate Eq. 14 forward from k0 with RK4. Returns (t, k) arrays."""
    Omega = omega(p) if Omega is None else Omega
    n_steps = int(T / dt)
    t = np.linspace(0.0, T, n_steps + 1)
    k = np.empty(n_steps + 1)
    k[0] = k0
    for i in range(n_steps):
        x = k[i]
        f1 = k_dot(x, p, Omega)
        f2 = k_dot(x + 0.5 * dt * f1, p, Omega)
        f3 = k_dot(x + 0.5 * dt * f2, p, Omega)
        f4 = k_dot(x + dt * f3, p, Omega)
        k[i + 1] = x + dt * (f1 + 2 * f2 + 2 * f3 + f4) / 6.0
    return t, k


# ---------------------------------------------------------------------------
# Health-capital steady state: the (H*, Y*) fixed point (Appendix B.4)
# ---------------------------------------------------------------------------

def health_map(H: float, p: Params, B: float, scale: float = 1.0) -> float:
    """The map T(H) = (eta0 + eta1*h - B * Y(H)) / delta_H  (B.4).

    Y(H) = y*(Omega(tau, H)) * scale, where scale stands in for the A*L
    level at the reference date, and B is proportional to chi*kappa*tau/Pop.
    T is strictly decreasing in H because Y(H) is increasing, so F(H) =
    T(H) - H has a single sign change and the positive fixed point is
    unique (the health-output loop is self-correcting negative feedback).
    """
    Y = y_star(p, omega(p, H=H)) * scale
    return (p.eta0 + p.eta1 * p.h - B * Y) / p.delta_H


def solve_health_fixed_point(
    p: Params, B: float, scale: float = 1.0, tol: float = 1e-12
) -> float:
    """Solve H* = T(H*) by bisection on F(H) = T(H) - H over H > 0."""
    f = lambda H: health_map(H, p, B, scale) - H
    lo, hi = 1e-9, 1.0
    while f(hi) > 0.0:
        hi *= 2.0
        if hi > 1e9:
            raise RuntimeError("no upper bracket found for the health fixed point")
    if f(lo) < 0.0:
        raise RuntimeError("no positive fixed point: T(0+) < 0")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


DEFAULT = Params()
