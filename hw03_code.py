
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


if __name__ == "__main__":
    # Default run for Part (a)
    problem1a(N=100)