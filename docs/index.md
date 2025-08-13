# Computational Math Toolkit (CMT)

A small, well-tested Python library of core numerical algorithms for applied mathematics (ODE/PDE solvers, linear algebra helpers, numerical analysis tools) with Jupyter demo notebooks and an interactive Streamlit demo.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Quickstart

```python
from cmt.ode import rk4
import numpy as np

f = lambda t, y: 0.5 * y
t, y = rk4(f, y0=1.0, t0=0.0, t1=2.0, dt=1e-3)
print(y[-1], np.exp(1.0))
```

## Modules

- ODE: `euler_forward`, `rk4`
- PDE: `solve_heat_explicit`, `solve_wave_1d`
- Linear Algebra: `svd_decompose`, `pca`, `eig_decompose`
- Analysis: `bisection`, `newton`, `simpson`, `linear_interpolate`

## Notebooks

- `notebooks/01_ode_examples.ipynb`
- `notebooks/02_pde_examples.ipynb`
- `notebooks/03_linear_algebra.ipynb`
- `notebooks/04_analysis.ipynb`

## Demo

Run the Streamlit demo locally:

```bash
streamlit run app/streamlit_app.py
```
