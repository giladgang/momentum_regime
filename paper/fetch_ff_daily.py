"""Fetch daily FF3 factors (public French library, no WRDS).
Usage: .venv/bin/python -m paper.fetch_ff_daily
Output: paper/results/data/ff_daily.parquet (mktrf, smb, hml, rf; decimals)
"""
import io
import os
import ssl
import sys
import urllib.request
import zipfile

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
from paper import config as C                                   # noqa: E402

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()

URL = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/'
       'ftp/F-F_Research_Data_Factors_daily_CSV.zip')


def main():
    req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    raw = urllib.request.urlopen(req, timeout=60, context=_SSL).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    csv = zf.read(zf.namelist()[0]).decode('latin1')
    lines = [ln for ln in csv.splitlines()]
    start = next(i for i, ln in enumerate(lines)
                 if ln.strip()[:8].isdigit())
    end = next((i for i in range(start, len(lines))
                if not lines[i].strip()[:8].isdigit()), len(lines))
    df = pd.read_csv(io.StringIO('\n'.join(lines[start:end])), header=None,
                     names=['date', 'mktrf', 'smb', 'hml', 'rf'])
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.set_index('date').astype(float) / 100.0
    os.makedirs(os.path.dirname(C.FF_DAILY), exist_ok=True)
    df.to_parquet(C.FF_DAILY)
    print(f'Saved {C.FF_DAILY}: {df.index.min().date()} -> '
          f'{df.index.max().date()} ({len(df)} days)')


if __name__ == '__main__':
    main()
