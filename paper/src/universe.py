"""Point-in-time top-N (by month-end market cap) universe membership."""


def top_n(stocks, n):
    df = stocks.sort_values(['date', 'me', 'permno'],
                            ascending=[True, False, True])
    rank = df.groupby('date').cumcount()
    return df[rank < n].sort_values(['date', 'permno']).reset_index(drop=True)


def cap_coverage(stocks, members):
    tot = stocks.groupby('date')['me'].sum()
    mem = members.groupby('date')['me'].sum()
    return (mem / tot).dropna()
