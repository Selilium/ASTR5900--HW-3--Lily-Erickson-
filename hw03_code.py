
"""
Lily Erickson
ASTR 5900 HW3: Numerical Integration involving Euler & Runge-Kutta Methods, and Maxwell-Boltzmann Distribution

Problem 1(a):
Solve dy/dx = y^2 + 1 with y(0)=0 using:
- Euler method
- RK4 method
Compare both to exact solution y = tan(x)
Using the same number of steps for each method.
"""

import numpy as np
import matplotlib.pyplot as plt

#================= PROBLEM 1(A) =======================================

# ----------------------------
# ODE definition
# dy/dx = f(x, y)
# ----------------------------
def f(x, y):
    return y**2 + 1.0


# Exact solution for comparison
def y_exact(x):
    return np.tan(x)


# ----------------------------
# Euler Method
# ----------------------------
def euler_method(f, x0, y0, x_end, N):
    x = np.linspace(x0, x_end, N + 1)
    h = (x_end - x0) / N
    y = np.zeros_like(x)
    y[0] = y0

    for i in range(N):
        y[i + 1] = y[i] + h * f(x[i], y[i])

    return x, y


# ----------------------------
# RK4 Method (4th-order Runge-Kutta)
# ----------------------------
def rk4_method(f, x0, y0, x_end, N):
    x = np.linspace(x0, x_end, N + 1)
    h = (x_end - x0) / N
    y = np.zeros_like(x)
    y[0] = y0

    for i in range(N):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + 0.5 * h, y[i] + 0.5 * h * k1)
        k3 = f(x[i] + 0.5 * h, y[i] + 0.5 * h * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x, y


# ----------------------------
# Problem 1(a) runner
# ----------------------------
def problem1a(x0=0.0, y0=0.0, x_end=1.3, N=100, prefix="problem1a"):
    """
    Generates:
    - <prefix>_solutions_N<N>.png
    - <prefix>_errors_N<N>.png
    Returns arrays for reuse in later parts.
    """
    # Solve with both methods (same N, same interval)
    xE, yE = euler_method(f, x0, y0, x_end, N)
    xR, yR = rk4_method(f, x0, y0, x_end, N)

    # Exact solution (same grid)
    yT = y_exact(xE)

    # --- Plot solutions ---
    plt.figure()
    plt.plot(xE, yT, label="Exact: tan(x)")
    plt.plot(xE, yE, label=f"Euler (N={N})")
    plt.plot(xR, yR, label=f"RK4 (N={N})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Problem 1(a): $dy/dx = y^2 + 1$, $y(0)=0$")
    plt.legend()
    plt.tight_layout()
    sol_path = f"{prefix}_solutions_N{N}.png"
    plt.savefig(sol_path, dpi=200)
    plt.close()

    # --- Plot absolute errors ---
    plt.figure()
    plt.plot(xE, np.abs(yE - yT), label="|Euler - exact|")
    plt.plot(xR, np.abs(yR - yT), label="|RK4 - exact|")
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("Absolute error")
    plt.title("Problem 1(a): Absolute Error (log scale)")
    plt.legend()
    plt.tight_layout()
    err_path = f"{prefix}_errors_N{N}.png"
    plt.savefig(err_path, dpi=200)
    plt.close()

    # Print summary (nice for sanity checks)
    print(f"[Problem 1(a)] Saved: {sol_path}, {err_path}")
    print(f"  Settings: x0={x0}, y0={y0}, x_end={x_end}, N={N}, h={(x_end-x0)/N:.6f}")
    print(f"  Final y at x={x_end}:")
    print(f"    Exact tan(x): {yT[-1]:.10f}")
    print(f"    Euler:        {yE[-1]:.10f}")
    print(f"    RK4:          {yR[-1]:.10f}")

    return xE, yE, yR, yT


#================= PROBLEM 1(B) =======================================
def problem1b(x0=0.0, y0=0.0, x_end=1.55, N=500, rel_thresh=0.01, prefix="problem1b"):
    """
    Part (b): Identify where numerical solution begins to break down.
    We define breakdown as first x where relative error exceeds rel_thresh.
    Saves:
      - <prefix>_rel_error.png
      - <prefix>_breakdown_table.txt
    """
    # Solve
    xE, yE = euler_method(f, x0, y0, x_end, N)
    xR, yR = rk4_method(f, x0, y0, x_end, N)

    # Exact and exact derivative
    yT = y_exact(xE)
    dydx_exact = 1.0 + yT**2  # since dy/dx = y^2 + 1

    # Relative errors (add tiny epsilon to avoid division issues near 0)
    eps = 1e-15
    rel_err_E = np.abs(yE - yT) / (np.abs(yT) + eps)
    rel_err_R = np.abs(yR - yT) / (np.abs(yT) + eps)

    # Find breakdown indices
    idxE = np.argmax(rel_err_E > rel_thresh) if np.any(rel_err_E > rel_thresh) else None
    idxR = np.argmax(rel_err_R > rel_thresh) if np.any(rel_err_R > rel_thresh) else None

    # Plot relative error vs x
    plt.figure()
    plt.plot(xE, rel_err_E, label="Euler rel. error")
    plt.plot(xR, rel_err_R, label="RK4 rel. error")
    plt.yscale("log")
    plt.axvline(np.pi/2, linestyle="--", label=r"$\pi/2$")
    plt.axhline(rel_thresh, linestyle="--", label=f"threshold = {rel_thresh}")
    plt.xlabel("x")
    plt.ylabel(r"Relative error  $|y_{\rm num}-y_{\rm exact}|/|y_{\rm exact}|$")
    plt.title("Problem 1(b): Onset of Numerical Breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_rel_error.png", dpi=200)
    plt.close()

    # Write a small table around the breakdown point(s)
    lines = []
    lines.append(f"Breakdown threshold (relative error) = {rel_thresh}\n")
    lines.append(f"x_end={x_end}, N={N}, h={(x_end-x0)/N}\n\n")
    lines.append("method, i, x, y_num, y_exact, rel_error, dydx_exact\n")

    def add_row(method, i):
        lines.append(
            f"{method}, {i}, {xE[i]:.6f}, { (yE[i] if method=='Euler' else yR[i]):.6e}, "
            f"{yT[i]:.6e}, { (rel_err_E[i] if method=='Euler' else rel_err_R[i]):.6e}, "
            f"{dydx_exact[i]:.6e}\n"
        )

    # Add a few rows near where things go bad (if found)
    if idxE is not None and idxE != 0:
        for j in range(max(0, idxE-3), min(len(xE), idxE+3)):
            add_row("Euler", j)
    else:
        lines.append("Euler did not exceed threshold in this range.\n")

    lines.append("\n")

    if idxR is not None and idxR != 0:
        for j in range(max(0, idxR-3), min(len(xE), idxR+3)):
            add_row("RK4", j)
    else:
        lines.append("RK4 did not exceed threshold in this range.\n")

    with open(f"{prefix}_breakdown_table.txt", "w") as f_out:
        f_out.writelines(lines)

    # Print summary to terminal
    if idxE is not None:
        print(f"[1(b)] Euler breakdown (rel err > {rel_thresh}) near x ≈ {xE[idxE]:.6f}")
    else:
        print(f"[1(b)] Euler did NOT exceed rel err > {rel_thresh} up to x_end={x_end}")

    if idxR is not None:
        print(f"[1(b)] RK4 breakdown (rel err > {rel_thresh}) near x ≈ {xE[idxR]:.6f}")
    else:
        print(f"[1(b)] RK4 did NOT exceed rel err > {rel_thresh} up to x_end={x_end}")

    print(f"[1(b)] Saved: {prefix}_rel_error.png and {prefix}_breakdown_table.txt")



#================= PROBLEM 1(C) =======================================
def problem1c(x0=0.0, y0=0.0, x_end=1.3,
              N_list=(25, 50, 100, 200, 400, 800),
              prefix="problem1c"):
    """
    Part (c): Convergence study.
    Show that decreasing step size h increases accuracy.
    Compare how error decreases for Euler vs RK4.
    Saves:
      - <prefix>_error_vs_h.png
      - <prefix>_convergence_table.txt
    """

    # Exact endpoint value
    yT_end = np.tan(x_end)

    rows = []
    rows.append("N, h, y_exact(x_end), y_Euler(x_end), abs_err_E, y_RK4(x_end), abs_err_RK4\n")

    h_vals = []
    err_E = []
    err_R = []

    for N in N_list:
        xE, yE = euler_method(f, x0, y0, x_end, N)
        xR, yR = rk4_method(f, x0, y0, x_end, N)

        h = (x_end - x0) / N
        eE = abs(yE[-1] - yT_end)
        eR = abs(yR[-1] - yT_end)

        h_vals.append(h)
        err_E.append(eE)
        err_R.append(eR)

        rows.append(f"{N}, {h:.8f}, {yT_end:.10f}, {yE[-1]:.10f}, {eE:.6e}, {yR[-1]:.10f}, {eR:.6e}\n")

    # Save table
    table_path = f"{prefix}_convergence_table.txt"
    with open(table_path, "w") as fout:
        fout.writelines(rows)

    # Plot error vs h on log-log scale
    plt.figure()
    plt.loglog(h_vals, err_E, marker="o", label="Euler: |error at x_end|")
    plt.loglog(h_vals, err_R, marker="o", label="RK4: |error at x_end|")
    plt.gca().invert_xaxis()  # optional: smaller h to the right feels intuitive to some people
    plt.xlabel("Step size h")
    plt.ylabel(r"Absolute error at $x_{\mathrm{end}}$")
    plt.title("Problem 1(c): Convergence (error vs step size)")
    plt.legend()
    plt.tight_layout()
    fig_path = f"{prefix}_error_vs_h.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    # Quick terminal summary
    print(f"[1(c)] Saved: {fig_path} and {table_path}")
    print("      (Expect Euler error ~ O(h), RK4 error ~ O(h^4) until round-off dominates.)")

    return np.array(h_vals), np.array(err_E), np.array(err_R)



#================= PROBLEM 1(D) =======================================
def problem1d(x0=0.0, y0=0.0, x_end=1.3,
              N_list=(25, 50, 100, 200, 400, 800, 1600, 3200),
              prefix="problem1d"):
    """
    Part (d): Convergence study WITHOUT using exact solution.
    Use highest-resolution case (largest N, smallest h) as reference truth.
    Plot fractional difference vs h on log-log scale for Euler and RK4.

    Saves:
      - <prefix>_fracdiff_vs_h.png
      - <prefix>_fracdiff_table.txt
    """

    N_ref = max(N_list)
    h_list = []
    frac_E = []
    frac_R = []

    # Reference ("truth") solutions at x_end
    _, yE_ref_arr = euler_method(f, x0, y0, x_end, N_ref)
    _, yR_ref_arr = rk4_method(f, x0, y0, x_end, N_ref)
    yE_ref = yE_ref_arr[-1]
    yR_ref = yR_ref_arr[-1]

    eps = 1e-15
    lines = []
    lines.append(f"Reference N_ref={N_ref}\n")
    lines.append("N, h, yE(x_end), yE_ref, fracdiff_E, yRK4(x_end), yRK4_ref, fracdiff_RK4\n")

    for N in N_list:
        h = (x_end - x0) / N
        xE, yE = euler_method(f, x0, y0, x_end, N)
        xR, yR = rk4_method(f, x0, y0, x_end, N)

        fdE = abs(yE[-1] - yE_ref) / (abs(yE_ref) + eps)
        fdR = abs(yR[-1] - yR_ref) / (abs(yR_ref) + eps)

        h_list.append(h)
        frac_E.append(fdE)
        frac_R.append(fdR)

        lines.append(
            f"{N}, {h:.8e}, {yE[-1]:.10f}, {yE_ref:.10f}, {fdE:.6e}, "
            f"{yR[-1]:.10f}, {yR_ref:.10f}, {fdR:.6e}\n"
        )

    # Save table
    table_path = f"{prefix}_fracdiff_table.txt"
    with open(table_path, "w") as fout:
        fout.writelines(lines)

    # Plot fractional difference vs h on log-log scale
    plt.figure()
    plt.loglog(h_list, frac_E, marker="o", label="Euler fractional diff (vs N_ref)")
    plt.loglog(h_list, frac_R, marker="o", label="RK4 fractional diff (vs N_ref)")
    plt.gca().invert_xaxis()  # optional; keeps smaller h to the right
    plt.xlabel("Step size h")
    plt.ylabel(r"Fractional difference at $x_{\mathrm{end}}$")
    plt.title("Problem 1(d): Convergence without exact solution")
    plt.legend()
    plt.tight_layout()
    fig_path = f"{prefix}_fracdiff_vs_h.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[1(d)] Saved: {fig_path} and {table_path}")
    print(f"      Reference cases: Euler(N_ref={N_ref}), RK4(N_ref={N_ref})")

    return np.array(h_list), np.array(frac_E), np.array(frac_R)


# ============================================================================
# Problem 2: Maxwell-Boltzmann
# ============================================================================

# Physical constants (SI)
kB = 1.380649e-23          # J/K
eV = 1.602176634e-19       # J
mH = 1.6735575e-27         # kg (hydrogen atom mass ~ proton mass)
T_star = 1.0e4             # K

# Hydrogen excitation energy n=1 -> n=2
dE_12 = 10.2 * eV          # J

def mb_speed_pdf(v, T=T_star, m=mH):
    """
    Maxwell-Boltzmann speed probability density f(v) for speeds (not velocity components):
    f(v) = 4*pi * (m/(2*pi*kT))^(3/2) * v^2 * exp(-m v^2 / (2 k T))
    Units: 1/(m/s) so that integral over dv gives dimensionless probability.
    """
    pref = 4.0 * np.pi * (m / (2.0 * np.pi * kB * T))**1.5
    return pref * v**2 * np.exp(-m * v**2 / (2.0 * kB * T))

# ----------------------------
# Numerical integrator 
# Composite Simpson's rule (with trapezoid fallback)
# ----------------------------
def integrate_simpson(func, a, b, N, **kwargs):
    """
    Composite Simpson's rule on [a,b] with N subintervals (N must be even).
    Returns approximate integral of func(v, **kwargs) dv.
    """
    if N < 2:
        raise ValueError("N must be >= 2")
    if N % 2 == 1:
        N += 1  # make it even

    h = (b - a) / N
    x = a + h * np.arange(N + 1)
    y = func(x, **kwargs)

    S = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2])
    return (h / 3.0) * S


def integrate_trap(func, a, b, N, **kwargs):
    """
    Composite trapezoid rule on [a,b] with N subintervals.
    """
    h = (b - a) / N
    x = a + h * np.arange(N + 1)
    y = func(x, **kwargs)
    return h * (0.5*y[0] + np.sum(y[1:-1]) + 0.5*y[-1])


def v_min_excitation(m=mH, dE=dE_12):
    """
    Minimum speed such that (1/2) m v^2 >= dE.
    """
    return np.sqrt(2.0 * dE / m)

#================= PROBLEM 1(A) =======================================
def problem2a(prefix="problem2a", T=T_star, m=mH):
    """
    (a) Plot MB speed distribution for hydrogen at T=10,000 K.
    Saves: figures/problem2a_mb_pdf.png
    """
    # characteristic speeds
    v_mp = np.sqrt(2.0 * kB * T / m)         # most probable speed
    v_rms = np.sqrt(3.0 * kB * T / m)        # rms speed
    v_mean = np.sqrt(8.0 * kB * T / (np.pi * m))  # mean speed

    v = np.linspace(0.0, 8.0*v_rms, 2000)
    f = mb_speed_pdf(v, T=T, m=m)

    plt.figure()
    plt.plot(v/1000.0, f, label=r"$f(v)$ for H at $T=10^4$ K")
    plt.axvline(v_mp/1000.0, linestyle="--", label=r"$v_{\rm mp}$")
    plt.axvline(v_mean/1000.0, linestyle="--", label=r"$\langle v\rangle$")
    plt.axvline(v_rms/1000.0, linestyle="--", label=r"$v_{\rm rms}$")
    plt.xlabel("Speed v (km/s)")
    plt.ylabel(r"Probability density $f(v)$ [s/m]")
    plt.title("Problem 2(a): Maxwell–Boltzmann Speed Distribution (Hydrogen)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{prefix}_mb_pdf.png", dpi=200)
    plt.close()

    print(f"[2(a)] Saved: figures/{prefix}_mb_pdf.png")
    print(f"      v_mp={v_mp/1000:.2f} km/s, v_mean={v_mean/1000:.2f} km/s, v_rms={v_rms/1000:.2f} km/s")

    return v_mp, v_mean, v_rms


def fraction_above_vmin(vmin, vmax, N, method="simpson", T=T_star, m=mH):
    """
    Compute integral_{vmin}^{vmax} f(v) dv using your integrator.
    """
    if method == "simpson":
        return integrate_simpson(mb_speed_pdf, vmin, vmax, N, T=T, m=m)
    elif method == "trap":
        return integrate_trap(mb_speed_pdf, vmin, vmax, N, T=T, m=m)
    else:
        raise ValueError("method must be 'simpson' or 'trap'")

#================= PROBLEM 1(B) =======================================
def problem2b(prefix="problem2b", T=T_star, m=mH, N=20000, vmax_factor=12.0, method="simpson"):
    """
    (b) Fraction of atoms fast enough to excite n=1 -> n=2.
    Integrate from vmin to 'infinity' approximated by vmax = vmax_factor * v_rms.
    Saves: tables/problem2b_result.txt
    """
    vmin = v_min_excitation(m=m, dE=dE_12)
    v_rms = np.sqrt(3.0 * kB * T / m)

    vmax = vmax_factor * v_rms
    frac = fraction_above_vmin(vmin, vmax, N, method=method, T=T, m=m)

    outpath = f"tables/{prefix}_result.txt"
    with open(outpath, "w") as f:
        f.write("Problem 2(b): Fraction above excitation threshold\n")
        f.write(f"T = {T:.1f} K\n")
        f.write(f"DeltaE(n=1->2) = 10.2 eV\n")
        f.write(f"v_min = {vmin:.6e} m/s ({vmin/1000:.3f} km/s)\n")
        f.write(f"v_max = {vmax:.6e} m/s ({vmax/1000:.3f} km/s) = {vmax_factor} * v_rms\n")
        f.write(f"Integrator = {method}, N = {N}\n")
        f.write(f"Fraction ≈ {frac:.12e}\n")

    print(f"[2(b)] Saved: {outpath}")
    print(f"      v_min={vmin/1000:.3f} km/s, vmax={vmax/1000:.3f} km/s, frac≈{frac:.3e}")

    return vmin, vmax, frac

#================= PROBLEM 1(C) =======================================
def problem2c(prefix="problem2c", T=T_star, m=mH,
              N_list=(2000, 5000, 10000, 20000, 40000),
              vmax_factors=(8, 10, 12, 14, 16),
              method="simpson"):
    """
    (c) Precision / error bar for (b):
        - convergence with step size (N)
        - convergence with vmax (approaching infinity)

    Saves:
      - figures/problem2c_convergence_N.png
      - figures/problem2c_convergence_vmax.png
      - tables/problem2c_convergence_summary.txt
    """
    vmin = v_min_excitation(m=m, dE=dE_12)
    v_rms = np.sqrt(3.0 * kB * T / m)

    # --- (1) Convergence with N at a fixed vmax ---
    vmax_fixed = max(vmax_factors) * v_rms
    frac_N = []
    for N in N_list:
        frac_N.append(fraction_above_vmin(vmin, vmax_fixed, N, method=method, T=T, m=m))
    frac_N = np.array(frac_N)

    plt.figure()
    plt.plot(N_list, frac_N, marker="o")
    plt.xlabel("Number of steps N")
    plt.ylabel("Fraction (v > v_min)")
    plt.title(f"Problem 2(c): Convergence with Step Size (vmax={max(vmax_factors)} v_rms)")
    plt.tight_layout()
    plt.savefig(f"figures/{prefix}_convergence_N.png", dpi=200)
    plt.close()

    # Estimate error from last two N values
    err_N = abs(frac_N[-1] - frac_N[-2])

    # --- (2) Convergence with vmax at fixed N ---
    N_fixed = max(N_list)
    frac_vmax = []
    vmax_vals = []
    for fac in vmax_factors:
        vmax = fac * v_rms
        vmax_vals.append(vmax)
        frac_vmax.append(fraction_above_vmin(vmin, vmax, N_fixed, method=method, T=T, m=m))
    frac_vmax = np.array(frac_vmax)
    vmax_vals = np.array(vmax_vals)

    plt.figure()
    plt.plot(vmax_factors, frac_vmax, marker="o")
    plt.xlabel(r"$v_{\max}/v_{\mathrm{rms}}$")
    plt.ylabel("Fraction (v > v_min)")
    plt.title(f"Problem 2(c): Convergence with vmax (N={N_fixed})")
    plt.tight_layout()
    plt.savefig(f"figures/{prefix}_convergence_vmax.png", dpi=200)
    plt.close()

    # Estimate error from last two vmax values
    err_vmax = abs(frac_vmax[-1] - frac_vmax[-2])

    # Conservative total error estimate
    err_total = np.sqrt(err_N**2 + err_vmax**2)

    outpath = f"tables/{prefix}_convergence_summary.txt"
    with open(outpath, "w") as f:
        f.write("Problem 2(c): Precision estimate for fraction above excitation threshold\n")
        f.write(f"T = {T:.1f} K\n")
        f.write(f"v_min = {vmin:.6e} m/s ({vmin/1000:.3f} km/s)\n")
        f.write(f"Integrator = {method}\n\n")

        f.write("Convergence with N (vmax fixed):\n")
        f.write(f"  vmax_fixed = {vmax_fixed:.6e} m/s = {max(vmax_factors)} * v_rms\n")
        for N, val in zip(N_list, frac_N):
            f.write(f"  N={N:6d}  frac={val:.12e}\n")
        f.write(f"  Step-size convergence estimate (|last - prev|) ~ {err_N:.3e}\n\n")

        f.write("Convergence with vmax (N fixed):\n")
        f.write(f"  N_fixed = {N_fixed}\n")
        for fac, val in zip(vmax_factors, frac_vmax):
            f.write(f"  vmax={fac:>4} v_rms  frac={val:.12e}\n")
        f.write(f"  vmax convergence estimate (|last - prev|) ~ {err_vmax:.3e}\n\n")

        f.write(f"Final recommended value (using largest N and largest vmax):\n")
        f.write(f"  frac ≈ {frac_vmax[-1]:.12e}\n")
        f.write(f"  conservative error bar ≈ {err_total:.3e}\n")

    print(f"[2(c)] Saved: figures/{prefix}_convergence_N.png, figures/{prefix}_convergence_vmax.png, {outpath}")
    print(f"      Recommended frac≈{frac_vmax[-1]:.3e} ± {err_total:.1e}")

    return frac_vmax[-1], err_total


if __name__ == "__main__":

    # ----- Problem 1(a) -----
    run_part_a = False
    if run_part_a:
        problem1a(N=100)

    # ----- Problem 1(b) -----
    run_part_b = False
    if run_part_b:
        problem1b(x_end=1.55, N=500, rel_thresh=0.01)

    # ----- Problem 1(c) -----
    run_part_c = False
    if run_part_c:
        problem1c(x_end=1.3, N_list=(25, 50, 100, 200, 400, 800))

    # ---- Problem 1(d) ----
    run_part_d = False
    if run_part_d:
        problem1d(x_end=1.3, N_list=(25, 50, 100, 200, 400, 800, 1600, 3200))

    # ----- Problem 2(a) -----
    run_2a = True
    if run_2a:
        problem2a()

    # ----- Problem 2(b) -----
    run_2b = True
    if run_2b:
        problem2b(N=20000, vmax_factor=12.0, method="simpson")

    # ----- Problem 2(c) -----
    run_2c = True
    if run_2c:
        problem2c()

    

        




 
