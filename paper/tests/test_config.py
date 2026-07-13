import os


def test_config_paths_and_constants():
    from paper import config as C
    assert os.path.exists(C.PANEL_EXT), C.PANEL_EXT
    assert os.path.exists(C.STOCK_EXT), C.STOCK_EXT
    assert os.path.exists(C.POOL_CSV)
    assert C.UNIVERSE_N == 1000
    assert C.EVAL_YEARS == list(range(2011, 2026))
    assert len(C.SEL_HMM_SEEDS) == 3 and C.SEL_HMM_SEEDS == [1, 2, 3]
    assert len(C.EVAL_HMM_SEEDS) == 50 and len(C.EVAL_XGB_SEEDS) == 20
    assert len(C.BIENNIAL_FOLDS) == 7 and len(C.ANNUAL_FOLDS) == 15
    assert C.VAL_END[7] == 2011 and C.VAL_END[101] == 2012 and C.VAL_END[114] == 2025
    assert C.VAL_END[115] == 2026     # live-2026 appendix evidence only
