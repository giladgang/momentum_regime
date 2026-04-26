"""
Lightweight WRDS diagnostic for UK + Japan Compustat Global coverage.

Aggregate-only queries (no full panel pulls). Verifies:
  1. Connection works.
  2. comp.g_company has SIC and GICS sector fields populated for GBR/JPN.
  3. Common-equity coverage by year, 1990-2025.
  4. Bank-flag feasibility (SIC 6000-6199 count by region/year).

Run-time target: under 60 seconds. No files written, output to stdout only.
"""
import sys
import wrds

print("Connecting to WRDS ...", flush=True)
db = wrds.Connection()
print("OK", flush=True)

def q(sql, label):
    print(f"\n--- {label} ---", flush=True)
    try:
        df = db.raw_sql(sql)
        print(df.to_string(index=False))
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return None

# 1. Schema check: does g_company have sic / gsector populated for our regions?
q("""
    SELECT loc,
           COUNT(*) AS n_rows,
           COUNT(sic) AS n_with_sic,
           COUNT(gsector) AS n_with_gsector
    FROM comp.g_company
    WHERE loc IN ('GBR','JPN')
    GROUP BY loc
    ORDER BY loc
""", "g_company schema: SIC / GICS sector populated for UK and JP?")

# 2. Common-equity security counts (gvkey-iid) per region
q("""
    SELECT sec.excntry,
           COUNT(DISTINCT (sec.gvkey || '_' || sec.iid)) AS n_securities
    FROM comp.g_security sec
    JOIN comp.g_company c ON sec.gvkey = c.gvkey
    WHERE sec.excntry IN ('GBR','JPN')
      AND sec.tpci = '0'
      AND c.loc = sec.excntry
    GROUP BY sec.excntry
    ORDER BY sec.excntry
""", "Total common-equity securities (loc = excntry, tpci='0')")

# 3. Annual security count, 1990-2025, both regions
q("""
    SELECT EXTRACT(YEAR FROM s.datadate)::int AS yr,
           c.loc,
           COUNT(DISTINCT (s.gvkey || '_' || s.iid)) AS n_active
    FROM comp.g_secd s
    JOIN comp.g_security sec ON s.gvkey = sec.gvkey AND s.iid = sec.iid
    JOIN comp.g_company  c   ON s.gvkey = c.gvkey
    WHERE c.loc IN ('GBR','JPN')
      AND sec.excntry = c.loc
      AND sec.tpci = '0'
      AND s.datadate BETWEEN '1990-01-01' AND '2025-12-31'
      AND s.prccd IS NOT NULL AND s.prccd > 0
      AND ((c.loc='GBR' AND s.curcdd='GBP') OR (c.loc='JPN' AND s.curcdd='JPY'))
    GROUP BY yr, c.loc
    ORDER BY yr, c.loc
""", "Active securities per year (with valid local-currency price)")

# 4. Bank count per region (any year) -- proves the BANK_REL flag will populate
q("""
    SELECT c.loc,
           COUNT(*) AS n_companies,
           COUNT(*) FILTER (WHERE sic BETWEEN '6000' AND '6199') AS n_banks_sic,
           COUNT(*) FILTER (WHERE gsector = '40')                AS n_financials_gics
    FROM comp.g_company c
    WHERE c.loc IN ('GBR','JPN')
    GROUP BY c.loc
    ORDER BY c.loc
""", "Bank / financials counts (SIC 6000-6199, GICS sector 40)")

db.close()
print("\nWRDS connection closed.", flush=True)
