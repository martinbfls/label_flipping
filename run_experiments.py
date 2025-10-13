import argparse
import os
import yaml
import json
import itertools
import time
import traceback
from types import SimpleNamespace
import pandas as pd
import re

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
    result = None
    try:
        if dry:
            print("DRY RUN: would call main with:", cfg)
            return 0
        result = run_main(cfg)
        print("-> finished successfully")
        time.sleep(wait)
        return 0, result
    except Exception as e:
        print("-> FAILED:", e)
        traceback.print_exc()
        return 1, result

def safe_filename(s):
    return re.sub(r'[^a-zA-Z0-9._\-]', '_', str(s))

def make_output_path(cfg):
    get = lambda *keys, default='NA': next((cfg[k] for k in keys if k in cfg and cfg[k] is not None), default)
    attack_method = safe_filename(get('attack_method', 'attack', default='NA'))
    loss_type = safe_filename(get('loss_type', 'loss', default='NA'))
    dataset = safe_filename(get('dataset', 'data', default='NA'))
    model = safe_filename(get('model', default='NA'))

    aggregation_method = safe_filename(get('aggregation_method', 'agg', 'aggregation', default='NA'))
    num_honest_workers = safe_filename(get('num_honest_workers', 'num_honest', 'n_honest', default='NA'))
    num_byzantine_worker = safe_filename(get('num_byzantine_worker', 'num_byzantine', 'n_byz', 'n_byzantine', default='NA'))
    controlled_subset_size = safe_filename(get('controlled_subset_size', 'controlled_subset', 'controlled_size', default='NA'))
    budget_ratio = safe_filename(get('budget_ratio', 'budget', default='NA'))
    byzantine_steps = safe_filename(get('byzantine_steps', 'byz_steps', 'byzantine_step', default='NA'))
    byzantine_lr = safe_filename(get('byzantine_lr', 'byz_lr', 'byzantine_learning_rate', default='NA'))
    adversarial_schedule = safe_filename(get('adversarial_scheduler', default='NA'))

    dir_path = os.path.join(
        "results_csv",
        attack_method,
        loss_type,
        dataset,
        model
    )

    filename = (
        f"agg-{aggregation_method}"
        f"_honest-{num_honest_workers}"
        f"_byzantine-{num_byzantine_worker}"
        f"_controlled-{controlled_subset_size}"
        f"_budget-{budget_ratio}"
        f"_steps-{byzantine_steps}"
        f"_lr-{byzantine_lr}"
        f"_adv_sch_{adversarial_schedule}.csv"
    )

    filename = safe_filename(filename)  # ensure file-safe
    return os.path.join(dir_path, filename)

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

    for i, cfg in enumerate(runs):
        print(f"\n[ {i+1}/{len(runs)} ]")
        defaults = cfg_file.get("defaults", {}) or {}
        merged = {**defaults, **cfg}

        ret, result = run_single(merged, dry=args.dry, wait=args.wait)

        rows = []
        for epoch_result in result:
            row = {"return": ret}
            row.update(epoch_result)
            rows.append(row)
        df = pd.DataFrame(rows)

        csv_path = make_output_path(merged)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        df.to_csv(csv_path, index=False)
        print(f"Saved results to: {csv_path}")

if __name__ == "__main__":
    main()
