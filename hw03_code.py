
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




#================= PROBLEM 1(E) =======================================




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
    run_part_c = True
    if run_part_c:
        problem1c(x_end=1.3, N_list=(25, 50, 100, 200, 400, 800))

    # ---- Problem 1(d) ----
    # ---- Problem 1(e) ----



 
