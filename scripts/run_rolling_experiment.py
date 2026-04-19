"""
Run the full pipeline with rolling-normalised HMM features.

All experiment outputs (tables, plots, CSVs, pickles) are collected into
experiments/rolling/ so that production artefacts are never left in a
dirty state.  After the run, production data is restored and tables are
regenerated from the production pickle.
"""
import subprocess, sys, os, shutil, glob

EXPERIMENT_DIR = 'experiments/rolling'

# ══════════════════════════════════════════════════════════════════════════════
# 1. Back up production data files
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  ROLLING EXPERIMENT -- isolated from production")
print("=" * 60)

PRODUCTION_BACKUPS = {
    'data/panel.parquet':              'data/panel_FIXED_BACKUP.parquet',
    'data/panel_with_regimes.parquet': 'data/panel_with_regimes_PRODUCTION.parquet',
    'cs_artefacts_data.pkl':           'cs_artefacts_data_PRODUCTION.pkl',
}

print("\nBacking up production data ...")
for src, dst in PRODUCTION_BACKUPS.items():
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  {src} -> {dst}")

# Snapshot which tables/plots already exist so we can tell what the
# experiment created or overwrote.
pre_tables = {f: os.path.getmtime(f) for f in glob.glob('tables/*.tex')}
pre_plots  = {f: os.path.getmtime(f)
              for f in glob.glob('*.png') + glob.glob('*.pdf') + glob.glob('*.csv')}

# ══════════════════════════════════════════════════════════════════════════════
# 2. Swap panel to rolling version and run pipeline
# ══════════════════════════════════════════════════════════════════════════════

print("\nSwapping to rolling-normalised panel ...")
import pandas as pd
panel_rolling = pd.read_parquet('data/panel_rolling.parquet')
panel_rolling.to_parquet('data/panel.parquet', index=False)

if os.path.exists('data/panel_with_regimes.parquet'):
    os.remove('data/panel_with_regimes.parquet')

print("\nRunning full pipeline with rolling normalisation ...")
print("This will take ~4-5 hours (HMM + XGB + all tables)\n")

steps = [1, 2, 3, 7, 8, 9, 10, 11, 13, 14]
failed = False
for step in steps:
    print(f"\n--- Step {step} ---")
    result = subprocess.run(
        [sys.executable, '-u', 'run_pipeline.py', '--step', str(step)],
        capture_output=False)
    if result.returncode != 0:
        print(f"Step {step} FAILED")
        failed = True
        break

# ══════════════════════════════════════════════════════════════════════════════
# 3. Collect ALL experiment outputs into experiments/rolling/
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nCollecting experiment outputs to {EXPERIMENT_DIR}/ ...")
os.makedirs(os.path.join(EXPERIMENT_DIR, 'tables'), exist_ok=True)

# Data artefacts
for src, dst_name in [
    ('data/panel_with_regimes.parquet', 'panel_with_regimes_ROLLING.parquet'),
    ('cs_artefacts_data.pkl',           'cs_artefacts_data_ROLLING.pkl'),
]:
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(EXPERIMENT_DIR, dst_name))

# Tables that were created or modified
for f in glob.glob('tables/*.tex'):
    shutil.copy2(f, os.path.join(EXPERIMENT_DIR, 'tables', os.path.basename(f)))

# Plots/CSVs that were created or modified
for f in glob.glob('*.png') + glob.glob('*.pdf') + glob.glob('*.csv'):
    mtime = os.path.getmtime(f)
    if f not in pre_plots or mtime > pre_plots[f]:
        shutil.copy2(f, os.path.join(EXPERIMENT_DIR, os.path.basename(f)))

# ══════════════════════════════════════════════════════════════════════════════
# 4. Restore production data
# ══════════════════════════════════════════════════════════════════════════════

print("\nRestoring production data ...")
for src, backup in PRODUCTION_BACKUPS.items():
    if os.path.exists(backup):
        shutil.copy2(backup, src)
        print(f"  {backup} -> {src}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Regenerate production tables from production pickle
#    This is the key fix: we don't just restore old table files (which might
#    themselves be stale), we regenerate from the authoritative source.
# ══════════════════════════════════════════════════════════════════════════════

print("\nRegenerating production tables (step 3) ...")
regen = subprocess.run(
    [sys.executable, '-u', 'run_pipeline.py', '--step', '3'],
    capture_output=False)
if regen.returncode != 0:
    print("ERROR: Production table regeneration failed!")
    print("Run manually:  python run_pipeline.py --step 3")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 6. Validate production integrity
# ══════════════════════════════════════════════════════════════════════════════

print("\nValidating production integrity ...")
validate = subprocess.run(
    [sys.executable, '-u', 'scripts/validate_production.py'],
    capture_output=True, text=True)
print(validate.stdout)
if validate.returncode != 0:
    if validate.stderr:
        print(validate.stderr)
    print("VALIDATION FAILED -- check production files!")
    sys.exit(1)

status = "with errors" if failed else "successfully"
print(f"\nRolling experiment completed {status}.")
print(f"Experiment outputs: {EXPERIMENT_DIR}/")
print("Production files restored and validated.")
