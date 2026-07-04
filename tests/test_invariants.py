"""
Metamorphic / property-based invariant tests for the directed-contagion pipeline.

WHY THIS EXISTS
    A solo author has no co-author to catch a silent implementation bug, and a green test
    suite that only checks a few hand-picked numbers proves almost nothing. These tests
    instead assert the STRUCTURAL INVARIANTS the math must satisfy for ANY valid input,
    generated automatically. A violation is a real bug in the load-bearing GFEVD / VAR /
    symmetrization-null code -- exactly the kind of coding artifact that could otherwise
    manufacture a spurious directedness result.

    The invariants encoded:
      * GFEVD rows sum to 1 and entries lie in [0, 1]            (variance-decomposition identity)
      * GFEVD / connectedness are label-permutation EQUIVARIANT  (relabelling nodes just relabels output)
      * net connectedness sums to zero                          (every unit's TO is another's FROM)
      * the non-negative VAR(1) surrogate has off-diagonals >= 0 (clean contagion kernel)
      * spectral radius scales linearly                          (criticality is well-defined)
      * the symmetrization null is exactly symmetric AND rho-matched (the null does what it claims)

    Run:  pytest tests/ -q          (uses Hypothesis if installed; degrades to seeded cases otherwise)
"""
import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pilot_cross_tier"))
from lsa_capstone import gfevd, connectedness, fit_var_nonneg, spectral_radius  # noqa: E402


def _load_func(path, name, glbls):
    """Compile ONE top-level function from a source file without executing the module body.
    (directedness_null.py runs a full analysis on import, so we extract just the function.)"""
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            exec(compile(mod, str(path), "exec"), glbls)
            return glbls[name]
    raise LookupError(f"{name} not found in {path}")


symmetrize_rho_matched = _load_func(
    ROOT / "pilot_review" / "directedness_null.py", "symmetrize_rho_matched", {"np": np}
)

# --------------------------------------------------------------------- builders
def _rng(seed):
    return np.random.default_rng(int(seed) % (2**32))

def make_phi(N, seed, rho=0.6):
    """Random VAR(1) coupling scaled to a stable spectral radius."""
    A = _rng(seed).standard_normal((N, N))
    r = spectral_radius(A)
    return A * (rho / r) if r > 0 else A

def make_sigma(N, seed):
    """Well-conditioned SPD innovation covariance."""
    B = _rng(seed + 7).standard_normal((N, N))
    return B @ B.T + N * np.eye(N)

def make_series(N, seed, T=80):
    """A positive multivariate stress series for surrogate calibration."""
    rng = _rng(seed + 13)
    M = np.abs(rng.standard_normal((T, N))).cumsum(0) / np.arange(1, T + 1)[:, None]
    return M + 1.0

def perm_of(N, seed):
    return _rng(seed + 101).permutation(N)


# --------------------------------------------------------------------- invariant checks (plain fns)
def check_rows_sum_to_one(N, seed):
    th = gfevd(make_phi(N, seed), make_sigma(N, seed), H=10)
    assert np.allclose(th.sum(axis=1), 1.0, atol=1e-9), th.sum(axis=1)

def check_entries_unit_interval(N, seed):
    th = gfevd(make_phi(N, seed), make_sigma(N, seed), H=10)
    assert th.min() >= -1e-12 and th.max() <= 1 + 1e-9, (th.min(), th.max())

def check_gfevd_permutation_equivariant(N, seed):
    Phi, Sig = make_phi(N, seed), make_sigma(N, seed)
    p = perm_of(N, seed)
    th = gfevd(Phi, Sig, H=10)
    th_p = gfevd(Phi[np.ix_(p, p)], Sig[np.ix_(p, p)], H=10)
    assert np.allclose(th_p, th[np.ix_(p, p)], atol=1e-9)

def check_net_sums_to_zero(N, seed):
    th = gfevd(make_phi(N, seed), make_sigma(N, seed), H=10)
    _TO, _FROM, NET, _tot = connectedness(th)
    assert abs(NET.sum()) < 1e-7, NET.sum()

def check_connectedness_permutation_equivariant(N, seed):
    th = gfevd(make_phi(N, seed), make_sigma(N, seed), H=10)
    p = perm_of(N, seed)
    TO, FROM, NET, tot = connectedness(th)
    TOp, FROMp, NETp, totp = connectedness(th[np.ix_(p, p)])
    assert np.allclose(NETp, NET[p], atol=1e-9)
    assert np.allclose(TOp, TO[p], atol=1e-9)
    assert np.isclose(totp, tot, atol=1e-9)

def check_nonneg_offdiagonal(N, seed):
    Phi, _c, _S = fit_var_nonneg(make_series(N, seed))
    off = Phi.copy()
    np.fill_diagonal(off, 0.0)
    assert off.min() >= -1e-6, off.min()   # bounded solver -> off-diagonals constrained >= 0

def check_spectral_radius_scaling(N, seed):
    Phi = make_phi(N, seed)
    s = 2.5
    assert np.isclose(spectral_radius(s * Phi), abs(s) * spectral_radius(Phi), rtol=1e-9)

def check_symmetrization_null(N, seed):
    Phi = make_phi(N, seed, rho=0.7)
    Ps = symmetrize_rho_matched(Phi)
    assert np.allclose(Ps, Ps.T, atol=1e-9), "null is not symmetric"
    off = Ps.copy(); np.fill_diagonal(off, 0.0)
    assert off.min() >= -1e-9, "null broke non-negativity"
    r0, rs = spectral_radius(Phi), spectral_radius(Ps)
    assert np.isclose(rs, r0, rtol=1e-6), (r0, rs)   # rho-matched to the original


CHECKS = [
    check_rows_sum_to_one,
    check_entries_unit_interval,
    check_gfevd_permutation_equivariant,
    check_net_sums_to_zero,
    check_connectedness_permutation_equivariant,
    check_nonneg_offdiagonal,
    check_spectral_radius_scaling,
    check_symmetrization_null,
]

# --------------------------------------------------------------------- test registration
try:
    from hypothesis import given, settings, strategies as st

    _dims = st.integers(min_value=2, max_value=6)
    _seeds = st.integers(min_value=0, max_value=2**31 - 1)

    def _mk(fn):
        @given(N=_dims, seed=_seeds)
        @settings(max_examples=150, deadline=None)
        def _t(N, seed):
            fn(N, seed)
        _t.__name__ = "test_" + fn.__name__[len("check_"):]
        return _t

    for _fn in CHECKS:
        globals()[_mk(_fn).__name__] = _mk(_fn)

except ImportError:  # fallback: seeded example sweep, still real coverage without Hypothesis
    import pytest

    @pytest.mark.parametrize("seed", range(120))
    @pytest.mark.parametrize("fn", CHECKS, ids=[f.__name__ for f in CHECKS])
    def test_invariant(fn, seed):
        fn(2 + seed % 5, seed)
