"""
train_energy_based_model.py
"""
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import argparse

from models import FCNet, ConvNet
from langevin import sample_langevin
from data import sample_2d_data

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=('8gaussians', '2spirals', 'checkerboard', 'rings', 'MNIST'))
parser.add_argument('model', choices=('FCNet', 'ConvNet'))
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate. default: 1e-3')
parser.add_argument('--stepsize', type=float, default=0.1, help='Langevin dynamics step size. default 0.1')
parser.add_argument('--n_step', type=int, default=100, help='The number of Langevin dynamics steps. default 100')
parser.add_argument('--n_epoch', type=int, default=100, help='The number of training epoches. default 100')
parser.add_argument('--alpha', type=float, default=1., help='Regularizer coefficient. default 100')
args = parser.parse_args()

# load dataset
N_train = 5000
N_val = 1000
N_test = 5000

X_train = sample_2d_data(args.dataset, N_train)
X_val = sample_2d_data(args.dataset, N_val)
X_test = sample_2d_data(args.dataset, N_test)

train_dl = DataLoader(TensorDataset(X_train), batch_size=32, shuffle=True, num_workers=8)
val_dl = DataLoader(TensorDataset(X_train), batch_size=32, shuffle=True, num_workers=8)
test_dl = DataLoader(TensorDataset(X_train), batch_size=32, shuffle=True, num_workers=8)

# build model
if args.model == 'FCNet':
    model = FCNet(in_dim=2, out_dim=1, l_hidden=(100, 100), activation='relu', out_activation='linear')
elif args.model == 'ConvNet':
    model = ConvNet(in_chan=1, out_chan=1)
model.cuda()

opt = Adam(model.parameters(), lr=args.lr)
    
# train loop
for i_epoch in range(args.n_epoch):
    l_loss = []
    for pos_x, in train_dl:
        
        pos_x = pos_x.cuda()
        
        neg_x = torch.randn_like(pos_x)
        neg_x = sample_langevin(neg_x, model, args.stepsize, args.n_steps, intermediate_samples=False)
        
        opt.zero_grad()
        pos_out = model(pos_x)
        neg_out = model(neg_x)
        
        loss = (pos_out - neg_out) + args.alpha * (pos_out ** 2 + neg_out ** 2)
        loss = loss.mean()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        opt.step()
        
        l_loss.append(loss.item())
    print(np.mean(l_loss))
        
        
    # draw samples
    
