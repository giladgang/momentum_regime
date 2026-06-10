from presentation import content as C


def test_canonical_headline_numbers():
    assert C.SHARPE_XGB == 1.11
    assert C.FF6_ALPHA == "24.1%"
    assert C.FF6_T == "4.81"
    assert C.PANIC_SHARPE == 1.53
    assert C.CALM_SHARPE == 0.84
    assert C.SHARPE_FIX12 == 0.06
    assert C.MDD_FIX12 == "-72.2%"
    assert C.CLUSTER_SHARPES == [0.92, 0.93, 1.21, 1.82]


def test_stale_gadi_numbers_are_not_used():
    # The old "call with Gadi" deck used a different cut; these must never appear.
    forbidden = ["24.7%", "1.56", "$15.7"]
    blob = " ".join(str(v) for v in vars(C).values() if isinstance(v, (str, int, float, list)))
    for bad in forbidden:
        assert bad not in blob, f"stale value {bad!r} present in content.py"
