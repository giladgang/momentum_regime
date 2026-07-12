import numpy as np

from paper.src import model as M


def test_ensemble_learns_and_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.01, size=500)
    s1 = M.ensemble_scores(X, y, X[:10], seeds=[1, 2], n_jobs=2)
    s2 = M.ensemble_scores(X, y, X[:10], seeds=[1, 2], n_jobs=2)
    np.testing.assert_array_equal(s1, s2)
    corr = np.corrcoef(s1, y[:10])[0, 1]
    assert corr > 0.9
