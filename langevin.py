import numpy as np
import torch
import torch.autograd as autograd


def sample_langevin(x, model, stepsize, n_steps, noise_scale=None, intermediate_samples=False):
    """Draw samples using Langevin dynamics
    x: torch.Tensor, initial points
    model: An energy-based model
    stepsize: float
    n_steps: integer
    noise_scale: Optional. float. If None, set to np.sqrt(stepsize * 2)
    """
    if noise_scale is None:
        noise_scale = np.sqrt(stepsize * 2)

    l_samples = []
    l_dynamics = []
    x.requires_grad = True
    for _ in range(n_steps):
        l_samples.append(x.detach().to('cpu'))
        noise = torch.randn_like(x) * noise_scale
        out = model(x)
        grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
        dynamics = stepsize * grad + noise
        x = x + dynamics
        l_samples.append(x.detach().to('cpu'))
        l_dynamics.append(dynamics.detach().to('cpu'))

    if intermediate_samples:
        return l_samples, l_dynamics
    else:
        return l_samples[-1]


