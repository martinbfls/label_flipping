import torch

class Worker:
    def __init__(self, model, loader, criterion, id):
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.id = id

    def get_batch(self):
        data, target = next(iter(self.loader))
        device = next(self.model.parameters()).device
        return data.to(device), target.to(device)
    
    def update_model(self, new_model):
        self.model.load_state_dict(new_model.state_dict())

    def get_gradient(self, plotting=False):
        data, target = self.get_batch()
        outputs = self.model(data)
        loss = self.criterion(outputs, target)
        grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=False)
        return [g.detach() for g in grads]