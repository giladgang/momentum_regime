"""Inspect Compustat Global g_secd schema for total-return inputs."""
import wrds
db = wrds.Connection()

print("=== g_secd: dividend / adjustment columns ===")
cols = db.describe_table('comp', 'g_secd')
keep = cols[cols['name'].str.contains(
    'div|adj|trf|aj|cheqv', case=False, regex=True
)]
print(keep.to_string(index=False))

print("\n=== g_secd: trfd (total-return factor) presence ===")
cols_trfd = cols[cols['name'].str.contains('trfd|trfm|tret', case=False, regex=True)]
print(cols_trfd.to_string(index=False) if len(cols_trfd) else "no trfd-like columns")

print("\n=== Sample of g_secd showing div + ajexdi for a UK firm with dividends ===")
sample = db.raw_sql("""
    SELECT s.gvkey, s.iid, s.datadate, s.prccd, s.ajexdi, s.div, s.divd, s.divsp, s.cheqv
    FROM comp.g_secd s
    WHERE s.gvkey = (
        SELECT s2.gvkey FROM comp.g_secd s2
        JOIN comp.g_company c ON s2.gvkey = c.gvkey
        WHERE c.loc='GBR' AND s2.div IS NOT NULL AND s2.div > 0
        LIMIT 1
    )
    AND s.div IS NOT NULL AND s.div > 0
    ORDER BY s.datadate
    LIMIT 5
""")
print(sample.to_string(index=False))

print("\n=== Coverage: % of UK and JP daily rows with non-null ajexdi and div ===")
cov = db.raw_sql("""
    SELECT c.loc,
           COUNT(*) AS n,
           ROUND(100.0 * COUNT(s.ajexdi) / COUNT(*), 2) AS pct_ajexdi,
           ROUND(100.0 * COUNT(s.div)    / COUNT(*), 2) AS pct_div
    FROM comp.g_secd s
    JOIN comp.g_company c ON s.gvkey = c.gvkey
    WHERE c.loc IN ('GBR','JPN')
      AND s.datadate BETWEEN '2020-01-01' AND '2024-12-31'
      AND s.prccd IS NOT NULL AND s.prccd > 0
    GROUP BY c.loc
    ORDER BY c.loc
""")
print(cov.to_string(index=False))

db.close()
