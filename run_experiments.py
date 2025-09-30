import argparse
import yaml
import json
import itertools
import time
import traceback
from types import SimpleNamespace

from main import main as run_main

def dict_to_namespace(d):
    return SimpleNamespace(**d)

def expand_grid(grid_dict):
    keys = list(grid_dict.keys())
    vals = [grid_dict[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def load_config(path):
    if path.endswith(".yml") or path.endswith(".yaml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise RuntimeError("Unsupported config format: use .yml/.yaml or .json")

def run_single(cfg, dry=False, wait=1.0):
    print(f"\n=== RUN: {cfg.get('attack_method','(no attack)')} | model={cfg.get('model_type')} | dataset={cfg.get('dataset')} ===")
    try:
        if dry:
            print("DRY RUN: would call main with:", cfg)
            return 0
        run_main(cfg)
        print("-> finished successfully")
        time.sleep(wait)
        return 0
    except Exception as e:
        print("-> FAILED:", e)
        traceback.print_exc()
        return 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='experiments.yml', help="Path to experiments.yml or .json")
    parser.add_argument("--dry", action="store_true", help="Don't actually run, just print")
    parser.add_argument("--wait", type=float, default=1.0, help="Seconds to wait between runs")
    parser.add_argument("--only", type=int, default=None, help="Run a single experiment index from the list/grid (0-based)")
    args = parser.parse_args()

    cfg_file = load_config(args.config)

    runs = []
    
    if cfg_file.get("experiments"):
        runs = cfg_file["experiments"]
    elif cfg_file.get("grid"):
        runs = list(expand_grid(cfg_file["grid"]))
    else:
        raise RuntimeError("Config must contain either 'experiments' (list) or 'grid' (dict).")

    if args.only is not None:
        runs = [runs[args.only]]

    results = []
    for i, cfg in enumerate(runs):
        print(f"\n[ {i+1}/{len(runs)} ]")
        defaults = cfg_file.get("defaults", {}) or {}
        merged = {**defaults, **cfg}
        ret = run_single(merged, dry=args.dry, wait=args.wait)
        results.append({"index": i, "cfg": merged, "return": ret})

    ok = sum(1 for r in results if r["return"]==0)
    fail = len(results) - ok
    print(f"\nDone: {ok} successful, {fail} failed out of {len(results)} runs.")
    
    with open("runs_summary.json", "w") as f:
        import json
        json.dump(results, f, indent=2)
    return

if __name__ == "__main__":
    main()
