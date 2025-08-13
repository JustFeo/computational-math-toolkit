import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from cmt.analysis import bisection, linear_interpolate, newton, simpson
from cmt.la import pca, svd_decompose
from cmt.ode import euler_forward, rk4
from cmt.pde import solve_heat_explicit, solve_wave_1d

st.set_page_config(page_title="CMT Demo", layout="wide")
st.title("Computational Math Toolkit — Demo")

module = st.sidebar.selectbox(
    "Module", ["ODE", "PDE", "Linear Algebra", "Analysis"], index=0
)

if module == "ODE":
    st.header("ODE: dy/dt = k*y")
    k = st.number_input("k", value=0.5)
    y0 = st.number_input("y0", value=1.0)
    t0 = st.number_input("t0", value=0.0)
    t1 = st.number_input("t1", value=2.0)
    dt = st.number_input("dt", value=1e-3, format="%g")
    method = st.selectbox("Method", ["RK4", "Euler Forward"])
    if st.button("Solve ODE"):
        f = lambda t, y: k * y
        if method == "RK4":
            t, y = rk4(f, y0=y0, t0=t0, t1=t1, dt=dt)
        else:
            t, y = euler_forward(f, y0=y0, t0=t0, t1=t1, dt=dt)
        y_analytic = y0 * np.exp(k * (t - t0))
        fig, ax = plt.subplots()
        ax.plot(t, y, label="numeric")
        ax.plot(t, y_analytic, "--", label="analytic")
        ax.set_xlabel("t")
        ax.legend()
        st.pyplot(fig)

elif module == "PDE":
    st.header("PDE: 1D Heat / Wave")
    pde_type = st.selectbox("Equation", ["Heat", "Wave"], index=0)
    n = st.slider("Grid points", min_value=51, max_value=401, value=201, step=10)
    dx = 1.0 / (n - 1)
    x = np.linspace(0.0, 1.0, n)
    steps = st.slider("Time steps", 10, 1000, 200, 10)
    if pde_type == "Heat":
        alpha = st.number_input("alpha", value=0.1)
        r = st.slider("r = alpha*dt/dx^2", min_value=0.01, max_value=0.49, value=0.25)
        dt = r * dx * dx / alpha
        u0 = np.exp(-200.0 * (x - 0.5) ** 2)
        if st.button("Run Heat"):
            U = solve_heat_explicit(u0, alpha=alpha, dx=dx, dt=dt, steps=steps)
            fig, ax = plt.subplots()
            for idx in np.linspace(0, steps, 5, dtype=int):
                ax.plot(x, U[idx], label=f"step {idx}")
            ax.set_title("Heat equation evolution (Neumann BCs)")
            ax.legend()
            st.pyplot(fig)
    else:
        c = st.number_input("c (wave speed)", value=1.0)
        r = st.slider("r = c*dt/dx", min_value=0.05, max_value=0.99, value=0.5)
        dt = r * dx / c
        u0 = np.sin(np.pi * x)
        v0 = np.zeros_like(u0)
        if st.button("Run Wave"):
            U = solve_wave_1d(u0, v0, c=c, dx=dx, dt=dt, steps=steps)
            fig, ax = plt.subplots()
            for idx in np.linspace(0, steps, 5, dtype=int):
                ax.plot(x, U[idx], label=f"step {idx}")
            ax.set_title("Wave equation evolution (Dirichlet BCs)")
            ax.legend()
            st.pyplot(fig)

elif module == "Linear Algebra":
    st.header("Linear Algebra: SVD / PCA")
    rng = np.random.default_rng(0)
    m = st.slider("m (rows)", 10, 200, 60)
    n = st.slider("n (cols)", 10, 200, 40)
    A = rng.normal(size=(m, n))
    U, S, Vt = svd_decompose(A)
    st.write("Top-5 singular values:", S[:5])
    n_components = st.slider("PCA components", 1, min(m, n), min(5, min(m, n)))
    components, var = pca(A, n_components=n_components)
    st.write("Explained variance:", var)

else:
    st.header("Analysis: Roots & Integration")
    st.subheader("Bisection vs Newton on cos(x)-x")
    a = 0.0
    b = 1.0
    f = lambda x: np.cos(x) - x
    df = lambda x: -np.sin(x) - 1.0
    root_b = bisection(f, a, b)
    root_n = newton(f, df, x0=0.5)
    st.write({"bisection": root_b, "newton": root_n})
    st.subheader("Simpson: ∫_0^π sin(x) dx")
    res = simpson(np.sin, 0.0, np.pi, n=1000)
    st.write("Result:", res)
