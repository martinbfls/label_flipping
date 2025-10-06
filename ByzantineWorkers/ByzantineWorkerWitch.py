from ByzantineWorkers.ByzantineWorker_ import ByzantineWorker_
import torch
import torch.nn.functional as F
import random
import math
from utils.utils import setup_optimizer, setup_scheduler
from utils.showing_results import logits_optimization

# logging.basicConfig(
#         level=logging.INFO, 
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         filename=config['log_file'],
#         filemode='w'
#     )

class ByzantineWorkerWitch(ByzantineWorker_):
    def __init__(self, model, loader, criterion, targeted_data, target_label, adversarial_label, save_path,
                 budget=5, controlled_subset_size=1.0, steps=5, lr=0.1, random_restart=10):
        super().__init__(model, loader, criterion, save_path, budget,
                          controlled_subset_size, steps, lr, random_restart)
        self.targeted_data = targeted_data
        self.target_label = target_label
        self.adversarial_label = adversarial_label

    def get_targeted_batch(self):
        data, target = next(iter(self.targeted_data))
        device = next(self.model.parameters()).device
        return data.to(device), target.to(device)

    def _optimize_logits(self, controlled_inputs, controlled_targets, clean_grads):
        device = next(self.model.parameters()).device
        num_classes = self.model(controlled_inputs).shape[1]
        flat_clean = torch.cat([g.flatten() for g in clean_grads])

        best_probs = None
        best_new_labels = None
        best_loss = float('inf')

        for restart in range(self.random_restart):
            logits = torch.randn(len(controlled_targets), num_classes, device=device, requires_grad=True)
            optimizer_logits = torch.optim.Adam([logits], lr=self.lr)
            scheduler = setup_scheduler(optimizer_logits, sched_type=self.scheduler, step_size=10, gamma=0.1) if self.scheduler else None
            attack_losses = []

            for _ in range(self.steps):
                pseudo_labels = F.softmax(logits, dim=1)
                preds = self.model(controlled_inputs)
                log_probs = F.log_softmax(preds, dim=1)
                loss_soft = -(pseudo_labels * log_probs).sum(dim=1).mean()

                adv_grads = torch.autograd.grad(loss_soft, self.model.parameters(), create_graph=True)
                flat_adv = torch.cat([g.flatten() for g in adv_grads])

                cos_sim = F.cosine_similarity(flat_clean.unsqueeze(0), flat_adv.unsqueeze(0))

                if self.targeted_data is not None and self.adversarial_label is not None:
                    attack_loss = 1 - cos_sim.mean()
                else:
                    attack_loss = cos_sim.mean()

                optimizer_logits.zero_grad()
                attack_loss.backward()
                optimizer_logits.step()
                attack_losses.append(attack_loss.item())
                
                if scheduler is not None:
                    scheduler.step()

                logits.data = torch.clamp(logits.data, -16.0, 16.0)

            with torch.no_grad():
                final_probs = F.softmax(logits, dim=1).detach()
                final_new_labels = final_probs.argmax(dim=1)
                
                if attack_loss.item() < best_loss:
                    best_loss = attack_loss.item()
                    best_probs = final_probs.clone()
                    best_new_labels = final_new_labels.clone()
                    best_attack_losses = attack_losses

        return best_probs, best_new_labels, best_attack_losses

    def _score_candidates(self, probs, new_labels, controlled_targets, clean_grads, controlled_inputs):
        device = probs.device
        sim_drop = []
        for j in range(len(controlled_targets)):
            tmp_targets = controlled_targets.clone()
            tmp_targets[j] = new_labels[j]
            preds_tmp = self.model(controlled_inputs[j:j+1])
            tmp_loss = self.criterion(preds_tmp, tmp_targets[j:j+1])
            adv_grad = torch.autograd.grad(tmp_loss, self.model.parameters(), create_graph=False)
            flat_adv = torch.cat([g.flatten() for g in adv_grad])
            flat_clean = torch.cat([g.flatten() for g in clean_grads])
            sim_drop.append(F.cosine_similarity(flat_clean.unsqueeze(0), flat_adv.unsqueeze(0)).item())
        score = -torch.tensor(sim_drop, device=device)
        return score

    def find_optimal_attack(self, data, targets, targeted_data, adversarial_label, plotting=False):
        device = next(self.model.parameters()).device
        data, targets = data.to(device), targets.to(device)

        self.model.train()
        targeted_grads = self._get_clean_gradients(targeted_data, adversarial_label)

        idx = random.sample(range(len(targets)), min(math.ceil(self.controlled_subset_size*len(targets)), len(targets)))
        controlled_inputs = data[idx]
        controlled_targets = targets[idx]
        
        probs, new_labels = self._optimize_logits(controlled_inputs, controlled_targets, targeted_grads)

        score = self._score_candidates(probs, new_labels, controlled_targets, targeted_grads, controlled_inputs)

        with torch.no_grad():
            topk = score.topk(min(self.budget, len(score))).indices
            attacked_targets = targets.clone()
            flips = 0
            for j in topk:
                if new_labels[j] != controlled_targets[j] and flips < self.budget:
                    attacked_targets[idx[j]] = new_labels[j]
                    flips += 1
        if plotting:
            logits_optimization(probs.cpu(), controlled_targets.cpu(), new_labels.cpu(), score.cpu(), self.save_path)

        return data, attacked_targets

    def get_gradient(self, plotting=False):
        data, target = self.get_batch()
        targeted_data, _targeted_label = self.get_targeted_batch()

        adversarial_label = torch.tensor([self.adversarial_label]*len(targeted_data), device=targeted_data.device)

        inputs, adv_targets = self.find_optimal_attack(data, target, targeted_data, adversarial_label, plotting=plotting)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, adv_targets)
        self.model.zero_grad()
        loss.backward()
        return [param.grad.clone().detach() for param in self.model.parameters()]