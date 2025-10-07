from ByzantineWorkers.ByzantineWorker_ import ByzantineWorker_
import torch
import torch.nn.functional as F
import random
import math
import logging
import copy
from utils.utils import setup_optimizer, setup_scheduler
from utils.showing_results import logits_optimization
import higher

class ByzantineWorkerGlobalTrajectoryMatching(ByzantineWorker_):
    def __init__(self, model, expert_model, loader, poisoned_loader, id, criterion, scheduler, save_path, budget=5,
                 controlled_subset_size=1.0, steps=5, lr=0.1, random_restart=10, num_classes=10, loss_type='l2'):
        super().__init__(model=model, loader=loader, criterion=criterion, id=id, 
                         scheduler=scheduler, save_path=save_path, budget=budget,
                         controlled_subset_size=controlled_subset_size, steps=steps, 
                         lr=lr, random_restart=random_restart, num_classes=num_classes)
        self.poisoned_loader = poisoned_loader
        self.poisoned_iter = iter(self.poisoned_loader)
        self.expert_model = expert_model
        self.loss_type = loss_type

    def get_controlled_batch(self):
        inputs, target = self.get_batch()
        n = len(target)
        k = min(math.ceil(self.controlled_subset_size * n), n)
        idx = random.sample(range(n), k)
        idx_tensor = torch.tensor(idx, dtype=torch.long, device=inputs.device if isinstance(inputs, torch.Tensor) else None)
        return inputs[idx_tensor], target[idx_tensor], idx_tensor

    def _get_poisoned_batch(self):
        try:
            data, target = next(self.poisoned_iter)
        except StopIteration:
            self.poisoned_iter = iter(self.poisoned_loader)
            data, target = next(self.poisoned_iter)
        device = next(self.model.parameters()).device
        return data.to(device), target.to(device)

    def train_higher(self, data, soft_labels, model, lr=0.01, steps=1):
        next_model = copy.deepcopy(model).to(next(model.parameters()).device)
        next_model.train()
        
        base_optimizer = torch.optim.SGD(next_model.parameters(), lr=lr)

        with higher.innerloop_ctx(next_model, base_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
            for _ in range(steps):
                output = fmodel(data)
                loss = self.criterion(output, soft_labels)
                diffopt.step(loss)

        return fmodel
    
    def train(self, data, target, model, lr=0.01, steps=1):
        next_model = copy.deepcopy(model).to(next(model.parameters()).device)
        next_model.train()
        
        optimizer = torch.optim.SGD(next_model.parameters(), lr=lr)

        for _ in range(steps):
            output = next_model(data)
            loss = self.criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return next_model

    def _get_model_params_vector(self, model):
        vecs = []
        for p in model.parameters():
            vecs.append(p.view(-1))
        if len(vecs) == 0:
            return torch.tensor([], device=next(model.parameters()).device)
        return torch.cat(vecs)

    def attack_loss(self, next_params_vec, current_params_vec, next_soft_vec, loss_type='l2', eps=1e-8):
        if loss_type == 'l2':
            num = torch.norm(next_params_vec - next_soft_vec, p=2)
            den = torch.norm(next_params_vec - current_params_vec, p=2)
            return num / (den + eps)
        elif loss_type == 'cosine_similarity':
            return 1 - F.cosine_similarity(next_params_vec.unsqueeze(0), next_soft_vec.unsqueeze(0)).mean()
        else:
            raise ValueError(f"Unknown loss type {loss_type}")

    def _optimize_logits(self, controlled_inputs, controlled_targets, num_classes):
        device = next(self.model.parameters()).device

        best_probs = None
        best_new_labels = None
        best_loss_val = float('inf')

        poisoned_inputs, poisoned_targets = self._get_poisoned_batch()

        next_expert_model = self.train(poisoned_inputs, poisoned_targets, self.expert_model, lr=0.01, steps=1)
        current_expert_vec = self._get_model_params_vector(self.expert_model)
        next_expert_vec = self._get_model_params_vector(next_expert_model)

        for restart in range(self.random_restart):
            logits = torch.randn(len(controlled_targets), num_classes, device=device, requires_grad=True)
            optimizer_logits = torch.optim.Adam([logits], lr=self.lr)
            scheduler = setup_scheduler(optimizer_logits, sched_type=self.scheduler, step_size=10, gamma=0.1) if self.scheduler else None
            attack_losses = []
            for _ in range(self.steps):
                soft_labels = F.softmax(logits, dim=1)

                fmodel = self.train_higher(controlled_inputs, soft_labels, self.expert_model, lr=0.01, steps=1)
                next_soft_vec = self._get_model_params_vector(fmodel)

                attack_loss_tensor = self.attack_loss(next_expert_vec, current_expert_vec, next_soft_vec, loss_type=self.loss_type)
                optimizer_logits.zero_grad()
                attack_loss_tensor.backward()
                optimizer_logits.step()

                attack_losses.append(attack_loss_tensor.item())

                if scheduler is not None:
                    scheduler.step()

                with torch.no_grad():
                    logits.clamp_(-16.0, 16.0)

            with torch.no_grad():
                final_probs = F.softmax(logits, dim=1).detach()
                final_new_labels = final_probs.argmax(dim=1)
                final_loss_val = attack_loss_tensor.item()

                if final_loss_val < best_loss_val:
                    best_loss_val = final_loss_val
                    best_probs = final_probs.clone()
                    best_new_labels = final_new_labels.clone()
                    best_attack_losses = attack_losses

        if best_probs is None:
            best_probs = F.softmax(torch.randn(len(controlled_targets), num_classes, device=device), dim=1)
            best_new_labels = best_probs.argmax(dim=1).clone()

        self.expert_model = next_expert_model

        return best_probs, best_new_labels, controlled_targets, best_attack_losses
    

    def _score_candidates(self, logits, y):
        if logits.ndim != 2:
            raise ValueError("logits must be 2D: (n_samples, n_classes)")
        n, C = logits.shape
        if y.shape[0] != n:
            raise ValueError("y must have length equal to number of rows in logits")
        mask = torch.arange(C, device=logits.device).unsqueeze(0).expand(n, -1)
        logits_masked = logits.masked_fill(mask == y.unsqueeze(1), -float("inf"))

        max_incorrect, _ = logits_masked.max(dim=1)
        scores = max_incorrect - logits[torch.arange(n, device=logits.device), y]
        return scores

    def find_optimal_attack(self, data, targets, plotting=False):
        controlled_inputs, controlled_targets, idx = self.get_controlled_batch()

        probs, new_labels, controlled_targets, attack_losses = self._optimize_logits(controlled_inputs, controlled_targets, self.num_classes)

        score = self._score_candidates(probs, controlled_targets)

        with torch.no_grad():
            k = min(self.budget, len(score))
            topk = score.topk(k).indices.cpu().tolist()
            attacked_targets = targets.clone()
            flips = 0
            for j in topk:
                if new_labels[j].item() != controlled_targets[j].item() and flips < self.budget:
                    attacked_targets[idx[j].item()] = new_labels[j]
                    flips += 1
        if plotting:
            logits_optimization(attack_losses, self.save_path.replace('.png', f'logits_optimization_byzantine_id_{self.id}_{self.count_plot}.png'))
            self.count_plot += 1

        return data, attacked_targets

    def get_gradient(self, plotting=False):
        data, target = self.get_batch()
        device = next(self.model.parameters()).device
        data, target = data.to(device), target.to(device)

        attacked_data, attacked_target = self.find_optimal_attack(data, target, plotting=plotting)

        self.model.train()

        output = self.model(attacked_data)
        loss = self.criterion(output, attacked_target)
        loss.backward()

        return [p.grad.detach().clone() for p in self.model.parameters()]