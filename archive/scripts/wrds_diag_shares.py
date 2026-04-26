"""Find the shares-outstanding column for Compustat Global."""
import wrds
db = wrds.Connection()

for table in ['g_fundq', 'g_funda', 'g_secd', 'g_security']:
    print(f"\n=== comp.{table} columns matching 'csh' or 'sho' ===")
    try:
        cols = db.describe_table('comp', table)
        keep = cols[cols['name'].str.contains('csh|sho|share', case=False, regex=True)]
        print(keep.to_string(index=False))
    except Exception as e:
        print(f"ERROR: {e}")

print("\n=== Sample row from g_secd showing share-like fields ===")
try:
    df = db.raw_sql("""
        SELECT *
        FROM comp.g_secd s
        JOIN comp.g_company c ON s.gvkey = c.gvkey
        WHERE c.loc = 'GBR'
        LIMIT 1
    """)
    share_cols = [c for c in df.columns if any(k in c.lower() for k in ['csh', 'sho', 'share'])]
    print("Share-like columns in g_secd:", share_cols)
    if share_cols:
        print(df[share_cols].to_string(index=False))
except Exception as e:
    print(f"ERROR: {e}")

db.close()
