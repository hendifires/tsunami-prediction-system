# scripts/smote_sweep.py
"""
SMOTE sweep runner
Usage:
  python scripts/smote_sweep.py --csv dataset/processed/volcanic_cleaned.csv --kfold 5 --out artifacts/smote_sweep --klist 3 5 7
"""
import argparse, subprocess, os, itertools, time, pathlib

SMOTE_VARIANTS = ['smote','smoteenn','smotetomek']

def run_one(csv, use_smote, variant, k, kfold, out_dir):
    run_name = f"{pathlib.Path(csv).stem}_{variant}_k{k}"
    outpath = os.path.join(out_dir, run_name)
    os.makedirs(outpath, exist_ok=True)
    cmd = [
        "python", "src/tsunami_prediction/run_experiment.py",
        "--csv", csv,
        "--use_smote",
        "--smote_variant", variant,
        "--kfold", str(kfold),
        "--out_dir", outpath
    ]
    # pass smote k via env variable for run_experiment to pick up (we'll read in run_experiment if present)
    env = os.environ.copy()
    if k is not None:
        env['SMOTE_K_NEIGHBORS'] = str(k)
    print("RUN:", " ".join(cmd), "env SMOTE_K_NEIGHBORS=", k)
    start = time.time()
    subprocess.run(cmd, check=True, env=env)
    print("Finished:", run_name, "in", time.time()-start, "s")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--kfold', type=int, default=5)
    p.add_argument('--out', default='artifacts/smote_sweep')
    p.add_argument('--klist', nargs='+', type=int, default=[3,5,7])
    p.add_argument('--variants', nargs='+', default=SMOTE_VARIANTS)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    for variant, k in itertools.product(args.variants, args.klist):
        run_one(args.csv, True, variant, k, args.kfold, args.out)
