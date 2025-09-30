# TODO : Code this abstract class
from Worker import Worker
import torch
import torch.nn.functional as F
import random
import math
import logging

class ByzantineWorker_(Worker):
    def __init__(self, model, loader, criterion, scheduler, budget=5,
                  controlled_subset_size=1.0, steps=5, lr=0.1, random_restart=10, num_classes=10):
        super().__init__(model, loader, criterion)
        self.budget = budget
        self.controlled_subset_size = controlled_subset_size
        self.steps = steps
        self.scheduler = scheduler
        self.lr = lr
        self.random_restart = random_restart
        self.num_classes = num_classes
    
    def _optimize_logits(self):
        pass

    def _score_candidates(self):
        pass

    def find_optimal_attack(self):
        pass

    def _get_clean_gradients(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        clean_grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        return [g.detach().clone() for g in clean_grads]

    def get_gradient(self):
        pass