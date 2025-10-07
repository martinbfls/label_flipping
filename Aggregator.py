import torch
import torch.optim as optim
import logging
import numpy as np
from utils.showing_results import plot_cta_pta

def test(model, test_loader, target_label):
        model.eval()
        correct = 0
        n_target = 0
        target_correct = 0
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                
                correct += pred.eq(target).sum().item()
                
                target_mask = (target == target_label)
                n_target += target_mask.sum().item()
                target_correct += pred[target_mask].eq(target[target_mask]).sum().item()

        target_acc = target_correct / n_target if n_target > 0 else None

        return correct / len(test_loader.dataset), target_acc

def evaluate_cta_pta(model, clean_test_loader, poisoned_test_loader):
    model.eval()
    device = next(model.parameters()).device

    correct_clean, total_clean = 0, 0
    correct_poison, total_poison = 0, 0

    with torch.no_grad():
        for x, y in clean_test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            correct_clean += (preds == y).sum().item()
            total_clean += y.size(0)

    cta = 100.0 * correct_clean / total_clean if total_clean > 0 else 0.0

    with torch.no_grad():
        for x, y in poisoned_test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            correct_poison += (preds == y).sum().item()
            total_poison += y.size(0)

    pta = 100.0 * correct_poison / total_poison if total_poison > 0 else 0.0

    return cta, pta

class Aggregator:
    def __init__(self, model, workers, optimizer, scheduler, save_path, aggregation_method="mean"):
        self.model = model
        self.workers = workers
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = save_path
        self.aggregation_method = aggregation_method

    def aggregate_gradients(self, grads_list):
        if self.aggregation_method == "mean":
            grads = [torch.mean(torch.stack(g), dim=0) for g in zip(*grads_list)]
        elif self.aggregation_method == "median":
            grads = [torch.median(torch.stack(g), dim=0).values for g in zip(*grads_list)]
        else:
            raise ValueError(f"Unknown aggregation method {self.aggregation_method}")
        return grads

    def train_round(self, plotting=False):
        grads_list = [w.get_gradient(plotting) for w in self.workers]
        grads = self.aggregate_gradients(grads_list)

        with torch.no_grad():
            for p, g in zip(self.model.parameters(), grads):
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                p.grad.copy_(g)
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        for w in self.workers:
            w.update_model(self.model)


    def train(self, test_loader, poisoned_test_loader, target_label, epochs=10, round_per_epoch=100):
        results = []
        cta_history, pta_history = [], []
        for epoch in range(epochs):
            k = torch.randint(0, 10*round_per_epoch, (1,)).item()
            for step in range(round_per_epoch):
                self.train_round(plotting=(step == k))
            
            acc, target_acc = test(self.model, test_loader, target_label)
            cta, pta = evaluate_cta_pta(self.model, test_loader, poisoned_test_loader)
            cta_history.append(cta)
            pta_history.append(pta)
            target_acc_str = f"{target_acc:.4f}" if target_acc is not None else "N/A"
            logging.info(f"Epoch {epoch+1}: Test accuracy = {acc:.4f}, Target label accuracy = {target_acc_str}, CTA = {cta:.4f}, PTA = {pta:.4f}")

            epoch_result = {
                'epoch': epoch + 1,
                'test_accuracy': acc,
                'target_accuracy': target_acc
            }
            results.append(epoch_result)
        plot_cta_pta(cta_history, pta_history, self.save_path)
        return results
