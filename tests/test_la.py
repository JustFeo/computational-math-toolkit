import numpy as np

from cmt.la import svd_decompose


def test_svd_reconstruction_error():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(50, 30))
    U, S, Vt = svd_decompose(A)
    A_rec = U @ np.diag(S) @ Vt
    err = np.linalg.norm(A - A_rec, ord="fro")
    assert err <= 1e-10
