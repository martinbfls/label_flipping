import torch
import torch.optim as optim
import logging

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

class Aggregator:
    def __init__(self, model, workers, optimizer, scheduler, aggregation_method="mean"):
        self.model = model
        self.workers = workers
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.aggregation_method = aggregation_method

    def aggregate_gradients(self, grads_list):
        if self.aggregation_method == "mean":
            grads = [torch.mean(torch.stack(g), dim=0) for g in zip(*grads_list)]
        elif self.aggregation_method == "median":
            grads = [torch.median(torch.stack(g), dim=0).values for g in zip(*grads_list)]
        else:
            raise ValueError(f"Unknown aggregation method {self.aggregation_method}")
        return grads

    def train_round(self):
        grads_list = [w.get_gradient() for w in self.workers]
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


    def train(self, test_loader, target_label, epochs=10, round_per_epoch=100):
        results = []
        
        for epoch in range(epochs):
            for step in range(round_per_epoch):
                self.train_round()
            
            acc, target_acc = test(self.model, test_loader, target_label)
            target_acc_str = f"{target_acc:.4f}" if target_acc is not None else "N/A"
            logging.info(f"Epoch {epoch+1}: Test accuracy = {acc:.4f}, Target label accuracy = {target_acc_str}")
            
            epoch_result = {
                'epoch': epoch + 1,
                'test_accuracy': acc,
                'target_accuracy': target_acc
            }
            results.append(epoch_result)
        
        return results
