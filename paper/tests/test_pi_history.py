import pandas as pd

from paper.build_pi_history import assemble


def test_assemble_prefers_walk_and_is_monotone():
    back = pd.DataFrame({
        'date': pd.to_datetime(['2010-11-30', '2010-12-31', '2011-01-31']),
        'pi': [0.2, 0.3, 0.99]})              # backfill overlaps 2011-01
    walk = pd.DataFrame({
        'date': pd.to_datetime(['2011-01-31', '2011-02-28']),
        'pi': [0.4, 0.5]})
    out = assemble(back, walk)
    assert list(out['src']) == ['backfill', 'backfill', 'walk', 'walk']
    assert out.loc[out['date'] == '2011-01-31', 'pi'].item() == 0.4
    assert out['date'].is_monotonic_increasing
    assert not out['date'].duplicated().any()
