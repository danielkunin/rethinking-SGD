import torch


# Extends SGD optimizer with added functionality
class SGD(torch.optim.SGD):
    def __init__(self, params, **opt_kwargs):
        super(SGD, self).__init__(params, **opt_kwargs)

        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["momentum_buffer"] = torch.zeros_like(p)

    def update(self, name, value):
        for group in self.param_groups:
            group[name] = value

    def reset(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["momentum_buffer"].mul_(0.0)

    @torch.no_grad()
    def velocity(self):
        velocity = []
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                else:
                    buf = torch.zeros_like(p)
                velocity.append(buf.reshape(-1))
        return torch.cat(velocity)

    def hyperparameters(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            break
        return lr, momentum
