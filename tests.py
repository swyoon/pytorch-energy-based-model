import torch
from models import FCNet
from langevin import sample_langevin

def test_langevin():
    X = torch.randn(100, 2)
    model = FCNet(2, 1, l_hidden=(50,))
    sample = sample_langevin(X, model, 0.1, 10)


