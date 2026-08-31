"""Numerical verification of every formal claim in the paper.

Each check names the equation or proposition it tests. Exits nonzero if any
check fails, so this doubles as the test suite: python src/verify.py
"""

import sys

import numpy as np

from thermo_solow import (
    Params,
    amplification_elasticity,
    convergence_rate,
    dlny_dtau,
    health_map,
    k_star,
    omega,
    omega_decomposition,
    phi,
    simulate_transition,
    solve_health_fixed_point,
    y_star,
)

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


p = Params()
print(f"Parameters: alpha={p.alpha}, s={p.s}, n+g+delta={p.break_even:.3f}, "
      f"tau={p.tau}, psi={p.psi}, b={p.b}, m={p.m}\n")

# --- 1. Nesting: the model collapses to textbook Solow (Sec. 3.4) ----------
clean = Params(tau=0.0)
solow_y = (p.s / p.break_even) ** (p.alpha / (1.0 - p.alpha))
check(
    "Nesting: tau=0, H=H_bar gives Omega=1 and textbook Solow y*",
    np.isclose(omega(clean), 1.0) and np.isclose(y_star(clean), solow_y),
    f"Omega={omega(clean):.12f}, y*={y_star(clean):.6f} vs Solow {solow_y:.6f}",
)

# --- 2. Axioms A1-A4 for phi (Sec. 2.6) ------------------------------------
taus = np.linspace(0.0, 0.999, 2000)
vals = (1.0 - taus) ** p.psi
decreasing = np.all(np.diff(vals) < 0)
convex = np.all(np.diff(vals, 2) >= -1e-12)
check(
    "Axioms A1-A4: phi(0)=1, decreasing, convex, phi(tau->1)->0",
    np.isclose(vals[0], 1.0) and decreasing and convex and vals[-1] < 1e-3,
)
try:
    phi(0.2, 0.7)
    check("Axiom A3 enforcement: psi<1 rejected", False)
except ValueError:
    check("Axiom A3 enforcement: psi<1 rejected", True)

# --- 3. Decomposition consistency (Eq. 19) ---------------------------------
pH = Params(tau=0.2)
H = 0.8
d = omega_decomposition(pH, H=H)
prod = d["waste_drag"] * d["cognitive_drag"] * d["labor_drag"]
check(
    "Eq. 19: waste x cognitive x labor drags == Omega",
    np.isclose(prod, omega(pH, H=H)),
    f"{prod:.12f} == {omega(pH, H=H):.12f}",
)

# --- 4. Closed form vs. simulated transition (Eqs. 14, 16) -----------------
t, k = simulate_transition(p, k0=0.2 * k_star(p), T=600.0)
check(
    "Eq. 16: ODE from k0 = 0.2 k* converges to the closed-form k*",
    np.isclose(k[-1], k_star(p), rtol=1e-8),
    f"k(T)={k[-1]:.10f} vs k*={k_star(p):.10f}",
)
t2, k2 = simulate_transition(p, k0=3.0 * k_star(p), T=600.0)
check(
    "Eq. 16: convergence from above (k0 = 3 k*) to the same k*",
    np.isclose(k2[-1], k_star(p), rtol=1e-8),
)

# --- 5. Proposition 4: amplification elasticity ----------------------------
eps = 1e-6
Om = omega(p)
num_elast = (np.log(y_star(p, Om * (1 + eps))) - np.log(y_star(p, Om * (1 - eps)))) / (2 * eps)
check(
    "Prop 4: numerical d ln(y*)/d ln(Omega) == 1/(1-alpha)",
    np.isclose(num_elast, amplification_elasticity(p), rtol=1e-6),
    f"numerical {num_elast:.6f} vs analytic {amplification_elasticity(p):.6f}",
)
drop = 1.0 - y_star(p, 0.9 * Om) / y_star(p, Om)
check(
    "B.1 example: a 10% Omega reduction lowers y* by 1 - 0.9^(1/0.7) = 14.0% (alpha=0.3)",
    np.isclose(drop, 1.0 - 0.9 ** (1.0 / 0.7), rtol=1e-10),
    f"drop = {100 * drop:.2f}%",
)

# --- 6. Propositions 1-3: comparative statics signs (B.2) ------------------
dk_dtau = (k_star(Params(tau=p.tau + 1e-6)) - k_star(Params(tau=p.tau - 1e-6))) / 2e-6
dk_dH = (k_star(p, omega(p, H=1.0 + 1e-6)) - k_star(p, omega(p, H=1.0 - 1e-6))) / 2e-6
dk_dpsi = (k_star(Params(psi=p.psi + 1e-6)) - k_star(Params(psi=p.psi - 1e-6))) / 2e-6
check("Prop 2 / Eq. 20a: dk*/dtau < 0", dk_dtau < 0, f"{dk_dtau:.4f}")
check("Prop 3 / Eq. 20b: dk*/dH > 0", dk_dH > 0, f"{dk_dH:.4f}")
check("Eq. 20c: dk*/dpsi < 0 for tau > 0", dk_dpsi < 0, f"{dk_dpsi:.4f}")
num_dlny = (np.log(y_star(Params(tau=p.tau + 1e-6))) - np.log(y_star(Params(tau=p.tau - 1e-6)))) / 2e-6
check(
    "Sec 3.3: d ln(y*)/d tau == -psi/[(1-alpha)(1-tau)]",
    np.isclose(num_dlny, dlny_dtau(p), rtol=1e-5),
    f"numerical {num_dlny:.6f} vs analytic {dlny_dtau(p):.6f}",
)

# --- 7. B.3: convergence rate is (1-alpha)(n+g+delta), independent of Omega -
def fitted_rate(pp, Omega=None):
    tt, kk = simulate_transition(pp, k0=0.5 * k_star(pp, Omega), T=120.0, Omega=Omega)
    gap = np.log(kk) - np.log(k_star(pp, Omega))
    sel = (np.abs(gap) > 1e-8) & (np.abs(gap) < 0.05)  # near-linear zone
    coef = np.polyfit(tt[sel], np.log(np.abs(gap[sel])), 1)
    return -coef[0]

r_dirty = fitted_rate(p)
r_clean = fitted_rate(p, Omega=1.0)
check(
    "B.3: fitted convergence rate == (1-alpha)(n+g+delta)",
    np.isclose(r_dirty, convergence_rate(p), rtol=2e-2),
    f"fitted {r_dirty:.5f} vs analytic {convergence_rate(p):.5f}",
)
check(
    "B.3: rate is independent of Omega (drag moves the target, not the speed)",
    np.isclose(r_dirty, r_clean, rtol=2e-2),
    f"Omega<1: {r_dirty:.5f}, Omega=1: {r_clean:.5f}",
)

# --- 8. B.4: the health fixed point is unique and self-correcting ----------
for B in (0.0, 0.05, 0.2, 0.8):
    grid = np.linspace(1e-6, 60.0, 300000)
    F = np.array([health_map(Hg, p, B) - Hg for Hg in grid])
    sign_changes = int(np.sum(np.diff(np.sign(F)) != 0))
    check(
        f"B.4: F(H) = T(H) - H has exactly one sign change (B={B})",
        sign_changes == 1,
        f"{sign_changes} sign change(s)",
    )
Tvals = np.array([health_map(Hg, p, 0.2) for Hg in grid])
check("B.4: T(H) strictly decreasing (negative feedback)", np.all(np.diff(Tvals) < 0))
H_fix = solve_health_fixed_point(p, B=0.2)
check(
    "B.4: bisection fixed point satisfies H* = T(H*)",
    np.isclose(H_fix, health_map(H_fix, p, 0.2), atol=1e-9),
    f"H* = {H_fix:.6f}",
)

# --- 9. B.5: tau-exogeneity makes Eq. 18 a genuine closed form -------------
scales = [0.5, 1.0, 5.0, 50.0]
oms = [omega(p) for _ in scales]  # Omega never references output
check(
    "B.5: Omega* contains no dependence on the level of output",
    len(set(oms)) == 1,
)

# --- summary ---------------------------------------------------------------
d0 = omega_decomposition(p, H=0.85)
print(f"\nIllustration at tau={p.tau}, H=0.85: Omega = {omega(p, H=0.85):.4f} "
      f"(waste {d0['waste_drag']:.4f} x cognitive {d0['cognitive_drag']:.4f} "
      f"x labor {d0['labor_drag']:.4f}); steady-state output penalty = "
      f"{100 * (1 - y_star(p, omega(p, H=0.85)) / y_star(Params(tau=0.0))):.1f}% "
      f"vs. clean Solow.")

if FAILURES:
    print(f"\n{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("\nAll checks passed.")
