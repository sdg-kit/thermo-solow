# Thermo-Solow

**Reference implementation and numerical verification for the paper _"Thermodynamic Waste, Health Capital, and Long-Run Growth: A Thermo-Ecological Augmentation of the Solow Model"_ (Greenlee, 2026).**

The paper ([PDF in `paper/`](paper/)) augments the Solow growth model with three coupled state variables absent from the standard framework: physical thermodynamic output (the waste every production process necessarily generates), health capital (linking pollution exposure to effective labor and TFP), and natural capital. Its central result is a closed-form steady state in which output per effective worker is the textbook Solow value multiplied by **Ω^(1/(1−α))**, where Ω = φ(τ)·ζ(H)·θ(H)^(1−α) is a thermo-ecological multiplier built from waste intensity and health. Because 1/(1−α) > 1, environmental degradation has an amplified effect on long-run output: it reduces production efficiency directly *and* depresses equilibrium capital accumulation. The model nests textbook Solow exactly (τ = 0, H = H̄ ⇒ Ω = 1), and generates a growth **drag** — a unique, stable, downward-shifted balanced growth path — not a poverty trap.

## What this repository contains

- `src/thermo_solow.py` — the core model, ~200 lines, numpy only. Every function cites its equation number in the paper: the sustainability multiplier φ(τ) = (1−τ)^ψ with the ψ ≥ 1 convexity axiom enforced (Eq. 9), the health channels θ and ζ (Eq. 7), the composite Ω and its three-way drag decomposition (Eqs. 11, 19), the fundamental dynamics (Eq. 14), the closed-form k* and y* (Eqs. 16, 18), analytic comparative statics (Eqs. 20a–c), the convergence rate (Appendix B.3), and the health-capital fixed point (Appendix B.4).
- `src/verify.py` — a numerical check of **every formal claim in the paper**, run as a script. It exits nonzero on any failure, so it doubles as the test suite.

## Verification

```
$ python src/verify.py
```

All 21 checks pass. Highlights:

```
[PASS] Nesting: tau=0, H=H_bar gives Omega=1 and textbook Solow y*
[PASS] Eq. 16: ODE from k0 = 0.2 k* converges to the closed-form k*  (k(T)=2.9054853273 vs k*=2.9054853273)
[PASS] Prop 4: numerical d ln(y*)/d ln(Omega) == 1/(1-alpha)  (numerical 1.428571 vs analytic 1.428571)
[PASS] B.1 example: a 10% Omega reduction lowers y* by 1 - 0.9^(1/0.7) = 14.0% (alpha=0.3)  (drop = 13.97%)
[PASS] Sec 3.3: d ln(y*)/d tau == -psi/[(1-alpha)(1-tau)]  (numerical -2.521008 vs analytic -2.521008)
[PASS] B.3: rate is independent of Omega (drag moves the target, not the speed)  (Omega<1: 0.05478, Omega=1: 0.05478)
[PASS] B.4: F(H) = T(H) - H has exactly one sign change (B=0.2)  (1 sign change(s))
```

The checks cover: exact nesting of textbook Solow; the four axioms A1–A4 on φ (including rejection of the concave ψ < 1 region); consistency of the Ω decomposition; agreement between the closed-form steady state and RK4-integrated transition paths from above and below; the amplification elasticity 1/(1−α) (Proposition 4); the signs of all comparative statics (Propositions 2–3, Eq. 20c) and the analytic derivative ∂ln y*/∂τ; the Appendix B.3 result that the convergence rate (1−α)(n+g+δ) is unchanged by the augmentation — the drag moves the target, not the speed; the uniqueness and self-correcting stability of the health-capital fixed point across a range of feedback strengths (Appendix B.4); and the τ-exogeneity that makes Eq. 18 a genuine closed form rather than an implicit equation (Appendix B.5).

## Usage

```python
from thermo_solow import Params, omega, y_star, omega_decomposition

p = Params(tau=0.15, psi=1.5)          # dirty economy, convex penalty
y_star(p) / y_star(Params(tau=0.0))    # steady-state output vs. clean Solow
omega_decomposition(p, H=0.85)         # waste / cognitive / labor drags
```

`Params` validates the paper's restrictions at construction: τ ∈ [0, 1), ψ ≥ 1 (axiom A3), α ∈ (0, 1).

## Scope and honesty notes

This repository implements the paper's *core analytic model* — the part the paper actually proves things about. It deliberately does **not** include the broader policy-simulation apparatus (carbon tax recycling, monetary policy, inequality dynamics) from an earlier phase of this project: that simulator was built on a superseded specification of the sustainability multiplier that the final paper explicitly rejects, and publishing it as a companion would misrepresent what the paper claims. The paper's empirical strategy (Section 6) and its listed companion-paper directions (Section 9) are future work.

## Citation

Greenlee, S. (2026). *Thermodynamic Waste, Health Capital, and Long-Run Growth: A Thermo-Ecological Augmentation of the Solow Model.* Working paper.

## License

Code: MIT. The paper (`paper/*.pdf`) is © the author; all rights reserved.
