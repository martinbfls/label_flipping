import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import os
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.metrics import auc as calc_auc

def logits_optimization(loss_values, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Attack loss over optimization steps')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Attack Loss')
    plt.title('Logits Optimization Process')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_cta_pta(cta_values, pta_values, cta_target_values, pta_target_values, cta_source_values, pta_source_values, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(cta_values, label='Clean Test Accuracy (CTA)')
    plt.plot(pta_values, label='Poisoned Test Accuracy (PTA)')
    plt.plot(cta_target_values, label='CTA on Target Class', linestyle='--')
    plt.plot(pta_target_values, label='PTA on Target Class', linestyle='--')
    plt.plot(cta_source_values, label='CTA on Source Class', linestyle=':')
    plt.plot(pta_source_values, label='PTA on Source Class', linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CTA and PTA over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path.replace('.png', 'cta_pta.png'))
    plt.close()

def plot_datasets_differences(clean_data, poisoned_data, save_path, source_label, target_label, inputs_or_labels='both', n_samples=5):

    def dataset_to_list(data):
        if hasattr(data, "__getitem__"):
            return list(data)
        else:
            samples = []
            for xb, yb in data:
                for i in range(len(yb)):
                    samples.append((xb[i], yb[i].item()))
            return samples

    clean_list = dataset_to_list(clean_data)
    poison_list = dataset_to_list(poisoned_data)

    # ---- Inputs ----
    if inputs_or_labels in ['both', 'inputs']:
        def get_examples(lst, label, n):
            return [x for x, y in lst if y == label][:n]

        clean_source = get_examples(clean_list, source_label, n_samples)
        poison_source = get_examples(poison_list, source_label, n_samples)
        clean_target = get_examples(clean_list, target_label, n_samples)
        poison_target = get_examples(poison_list, target_label, n_samples)

        # Messages si pas assez d'exemples
        if len(poison_source) < n_samples:
            print(f"[WARNING] Requested {n_samples} samples for poisoned source, but only found {len(poison_source)}")
        if len(poison_target) < n_samples:
            print(f"[WARNING] Requested {n_samples} samples for poisoned target, but only found {len(poison_target)}")

        fig, axes = plt.subplots(2, 2*n_samples, figsize=(3*2*n_samples, 6))

        def show(ax, tensor, title=None):
            if isinstance(tensor, torch.Tensor):
                arr = tensor.numpy()
            else:
                arr = np.array(tensor)
            if arr.ndim == 3 and arr.shape[0] in [1,3]:
                arr = np.transpose(arr, (1,2,0))
            # clamp automatique dans [0,1]
            arr = np.clip(arr, 0, 1)
            ax.imshow(arr, cmap="gray" if arr.ndim==2 or arr.shape[2]==1 else None)
            if title:
                ax.set_title(title)
            ax.axis("off")

        for i in range(min(n_samples, len(clean_source), len(poison_source))):
            show(axes[0, 2*i], clean_source[i], f"Clean src {source_label}")
            show(axes[0, 2*i+1], poison_source[i], f"Poisoned src {source_label}")

        for i in range(min(n_samples, len(clean_target), len(poison_target))):
            show(axes[1, 2*i], clean_target[i], f"Clean tgt {target_label}")
            show(axes[1, 2*i+1], poison_target[i], f"Poisoned tgt {target_label}")

        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_inputs.png'))
        plt.close()

    # ---- Labels ----
    if inputs_or_labels in ['both', 'labels']:
        clean_labels = [y for _,y in clean_list]
        poisoned_labels = [y for _,y in poison_list]

        plt.figure(figsize=(10, 6))
        sns.histplot(clean_labels, color='blue', label='Clean Data Labels', kde=False, stat='count')
        sns.histplot(poisoned_labels, color='red', label='Poisoned Data Labels', kde=False, stat='count')
        plt.title('Label Distribution Comparison')
        plt.xlabel('Labels')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(save_path.replace('.png', '_labels.png'))
        plt.close()

def plot_results_budget_differences(
    attack_method, attack_loss, dataset, scheduler, steps, lr, adv_scheduler,
    num_honest_workers, num_byzantine_workers, save_path
):
    controlled_subset_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    budgets = [0.01, 0.05, 0.1, 0.2, 0.3]
    metrics = [
        "final_cta", "final_pta",
        "final_cta_target", "final_pta_target",
        "final_cta_source", "final_pta_source"
    ]
    
    results = {
        css: {metric: [] for metric in metrics}
        for css in controlled_subset_sizes
    }

    for css in controlled_subset_sizes:
        for budget in budgets:
            file_path = (
                f"results_csv/{attack_method}/{attack_loss}/{dataset}/{scheduler}/"
                f"agg-mean_honest-{num_honest_workers}_byzantine-{num_byzantine_workers}_"
                f"controlled-{css}_budget-{budget}_steps-{steps}_lr-{lr}_adv_sch_{adv_scheduler}.csv"
            )
            if not os.path.exists(file_path):
                print(f"[WARNING] Missing file: {file_path}")
                continue
            try:
                data = pd.read_csv(file_path)
                results[css]["final_cta"].append((budget, data["clean_test_accuracy"].iloc[-1]))
                results[css]["final_pta"].append((budget, data["poisoned_test_accuracy"].iloc[-1]))
                results[css]["final_cta_target"].append((budget, data["clean_test_accuracy_target"].iloc[-1]))
                results[css]["final_pta_target"].append((budget, data["poisoned_test_accuracy_target"].iloc[-1]))
                results[css]["final_cta_source"].append((budget, data["clean_test_accuracy_source"].iloc[-1]))
                results[css]["final_pta_source"].append((budget, data["poisoned_test_accuracy_source"].iloc[-1]))
                print(f"[INFO] Loaded data from {file_path}")
            except Exception as e:
                print(f"[WARNING] Could not load {file_path}: {e}")

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for css in controlled_subset_sizes:
            if not results[css][metric]:
                continue
            
            sorted_points = sorted(results[css][metric], key=lambda x: x[0])
            x_vals, y_vals = zip(*sorted_points)
            plt.plot(x_vals, y_vals, marker="o", label=f"CSS={css}")

        plt.title(f"{metric.replace('_', ' ').upper()} vs Budget")
        plt.xlabel("Budget")
        plt.ylabel(metric.replace('final_', '').replace('_', ' ').title())
        plt.legend(title="Controlled subset size")
        plt.grid(True)
        plt.tight_layout()

        metric_path = os.path.join(save_path, f"{metric}_vs_budget.png")
        plt.savefig(metric_path)
        print(f"[INFO] Saved plot: {metric_path}")
        plt.close()

def plot_css_differences(
    attack_method, attack_loss, dataset, scheduler, steps, lr, adv_scheduler,
    num_honest_workers, num_byzantine_workers, save_path
):
    controlled_subset_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    budget = 1.0
    metrics = [
        "final_cta", "final_pta",
        "final_cta_target", "final_pta_target",
        "final_cta_source", "final_pta_source"
    ]
    
    results = {metric: [] for metric in metrics}

    for css in controlled_subset_sizes:
        file_path = (
            f"results_csv/{attack_method}/{attack_loss}/{dataset}/{scheduler}/"
            f"agg-mean_honest-{num_honest_workers}_byzantine-{num_byzantine_workers}_"
            f"controlled-{css}_budget-{budget}_steps-{steps}_lr-{lr}_adv_sch_{adv_scheduler}.csv"
        )
        if not os.path.exists(file_path):
            print(f"[WARNING] Missing file: {file_path}")
            continue
        try:
            data = pd.read_csv(file_path)
            results["final_cta"].append((css, data["clean_test_accuracy"].iloc[-1]))
            results["final_pta"].append((css, data["poisoned_test_accuracy"].iloc[-1]))
            results["final_cta_target"].append((css, data["clean_test_accuracy_target"].iloc[-1]))
            results["final_pta_target"].append((css, data["poisoned_test_accuracy_target"].iloc[-1]))
            results["final_cta_source"].append((css, data["clean_test_accuracy_source"].iloc[-1]))
            results["final_pta_source"].append((css, data["poisoned_test_accuracy_source"].iloc[-1]))
            print(f"[INFO] Loaded data from {file_path}")
        except Exception as e:
            print(f"[WARNING] Could not load {file_path}: {e}")

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        if not results[metric]:
            continue
        
        sorted_points = sorted(results[metric], key=lambda x: x[0])
        x_vals, y_vals = zip(*sorted_points)
        plt.plot(x_vals, y_vals, marker="o")
        plt.title(f"{metric.replace('_', ' ').upper()} vs Controlled Subset Size")
        plt.xlabel("Controlled Subset Size")
        plt.ylabel(metric.replace('final_', '').replace('_', ' ').title())
        plt.grid(True)
        plt.tight_layout()

        metric_path = os.path.join(save_path, f"{metric}_vs_css.png")
        plt.savefig(metric_path)
        print(f"[INFO] Saved plot: {metric_path}")
        plt.close()
