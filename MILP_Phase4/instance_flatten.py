import os, shutil
import pandas as pd
from ast import literal_eval


# STEP 1 — Build instances_flat/ from the per-problem subfolders

os.makedirs('instances_flat', exist_ok=True)
copied = 0
missing_dirs = []
for prob in ['CA', 'MIS', 'MVC']:
    src = f'instances/{prob}/train'
    if not os.path.isdir(src):
        missing_dirs.append(src)
        continue
    for f in os.listdir(src):
        if not f.endswith('.lp'):
            continue
        dst = f'instances_flat/{f}'
        if not os.path.exists(dst):
            shutil.copy2(f'{src}/{f}', dst)
            copied += 1

total_instances = len(os.listdir('instances_flat'))
print(f"[STEP 1] Copied {copied} new instance files.")
print(f"         instances_flat/ now has {total_instances} files.")
if missing_dirs:
    print(f"         WARNING: missing source dirs: {missing_dirs}")


# STEP 2 — Verify your backdoor CSVs are in the right place

csv_dir = 'data/backdoor_results'
print(f"\n[STEP 2] Checking {csv_dir}...")
if not os.path.isdir(csv_dir):
    print(f"         ERROR: {csv_dir} doesn't exist!")
    print(f"         Make sure your backdoor CSVs are at exactly this path.")
else:
    csvs = [f for f in os.listdir(csv_dir) if os.path.isfile(f'{csv_dir}/{f}')]
    print(f"         Found {len(csvs)} files in {csv_dir}")
    by_prob = {p: sum(1 for f in csvs if f.startswith(p + '_')) for p in ['CA', 'MIS', 'MVC']}
    for p, n in by_prob.items():
        print(f"         {p}: {n}")


# STEP 3 — Verify each CSV has a matching instance file

print(f"\n[STEP 3] Checking pairing between CSVs and instance files...")
unpaired = [f for f in csvs if not os.path.exists(f'instances_flat/{f}')]
print(f"         Unpaired CSVs (would crash training): {len(unpaired)}")
if unpaired:
    print(f"         First 5: {unpaired[:5]}")


# STEP 4 — Verify CSV format and row counts

print(f"\n[STEP 4] Verifying CSV contents...")
short = []
malformed = []
for f in csvs:
    p = f'{csv_dir}/{f}'
    try:
        df = pd.read_csv(p, index_col=0)
        if 'backdoor_list' not in df.columns or 'run_time' not in df.columns:
            malformed.append((f, f"wrong columns: {list(df.columns)}"))
            continue
        if len(df) < 10:
            short.append((f, len(df)))
            continue
        # Verify the first backdoor_list is parseable as a Python list
        literal_eval(df.iloc[0]['backdoor_list'])
    except Exception as e:
        malformed.append((f, str(e)[:60]))

print(f"         CSVs with <10 rows (would crash training): {len(short)}")
for f, n in short[:5]:
    print(f"           {f}: {n} rows")
print(f"         Malformed CSVs: {len(malformed)}")
for f, err in malformed[:5]:
    print(f"           {f}: {err}")


# FINAL SUMMARY

print("\n" + "=" * 60)
ready = (
    total_instances > 0
    and len(csvs) > 0
    and len(unpaired) == 0
    and len(short) == 0
    and len(malformed) == 0
)
if ready:
    print("✅ Backdoor data is correctly structured for MULTITASK training.")
    print(f"\nUse these flags in the MULTITASK command:")
    print(f"   --bd_instance_dir instances_flat")
    print(f"   --bd_result_dir   {csv_dir}")
else:
    print("❌ Issues above must be fixed before training will work.")
    print("   Tell me what the output says and I'll help fix each one.")