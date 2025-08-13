# Computational Math Toolkit (CMT)

A small, well-tested Python library of core numerical algorithms for applied mathematics (ODE/PDE solvers, linear-algebra helpers, numerical analysis tools) with Jupyter demo notebooks and an interactive Streamlit demo. Packagable to PyPI, documented with MkDocs, and CI/CD via GitHub Actions. MIT license.

## Install (dev)
```bash
git clone https://github.com/JustFeo/computational-math-toolkit.git
cd computational-math-toolkit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick demo
```python
from cmt.ode import rk4
def f(t, y): return 0.5*y
t, y = rk4(f, y0=1.0, t0=0.0, t1=2.0, dt=1e-3)
```

## Run tests
```bash
pytest
```

## Run demo app
```bash
streamlit run app/streamlit_app.py
```

## Documentation
- [Docs (MkDocs)](docs/index.md)
- [Demo Notebooks](notebooks/)
- [Streamlit Demo](app/streamlit_app.py)

---

See the project specification in the repo for full details, modules, and acceptance criteria.
