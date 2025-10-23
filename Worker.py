import torch

class Worker:
    def __init__(self, model, loader, criterion, id, device=None):
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.id = id
        self._loader_iter = iter(self.loader)
        self.device = device if device is not None else next(self.model.parameters()).device

    def get_batch(self):
        try:
            data, target = next(self._loader_iter)
        except StopIteration:
            # restart the iterator
            self._loader_iter = iter(self.loader)
            data, target = next(self._loader_iter)
        device = self.device
        return data.to(device), target.to(device)
    
    def update_model(self, new_model):
        self.model.load_state_dict(new_model.state_dict())

    def get_gradient(self, plotting=False):
        data, target = self.get_batch()
        outputs = self.model(data)
        loss = self.criterion(outputs, target)
        grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=False)
        return [g.detach() for g in grads]
    
# We can use this if not all parameters are used:grads = torch.autograd.grad(loss, tuple(self.model.parameters()), retain_graph=False, allow_unused=True)
# grads_fixed = [g.detach().clone() if g is not None else torch.zeros_like(p)
            #    for g, p in zip(grads, self.model.parameters())]
# return grads_fixed
# 