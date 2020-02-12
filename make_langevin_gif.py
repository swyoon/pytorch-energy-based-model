"""
make_langevin_gif.py
===================
Generate Langevin dyanmics sampling gif file using matplotlib
Output files are saved in imgs directory
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
import argparse

from langevin import sample_langevin
from data import sample_2d_data, potential_fn


parser = argparse.ArgumentParser()
parser.add_argument('energy_function', help='select toy energy function to generate sample from. (u1, u2, u3, u4)',
                    choices=['u1', 'u2', 'u3', 'u4'])
parser.add_argument('--no-arrow', action='store_true', help='disable display of arrows')
parser.add_argument('--out', help='the name of output file. default is the name of energy function.  ex) u1.gif',
                    default=None)
args = parser.parse_args()

def init():
    """initialize animation"""
    global point, arrow
    ax = plt.gca()
    ax.contour(XX, YY, np.exp(-e_grid.view(100,100)))
    return (point, arrow)


def update(i):
    """update animation for i-th frame"""
    global point, arrow, ax
    g = l_dynamics[i]
    s = l_sample[i]

    point.set_offsets(s)
    arrow.set_offsets(s)
    arrow.set_UVC(U=g[:,0], V=g[:,1])
    ax.set_title(f'Step: {i}')
    return (point, arrow)

# configuration
grid_lim = 4
n_grid = 100
n_sample = 100

stepsize = 0.03
n_steps = 100

# prepare for contour plot
energy_fn = potential_fn(args.energy_function)

xs = np.linspace(- grid_lim, grid_lim, n_grid)
ys = np.linspace(- grid_lim, grid_lim, n_grid)
XX, YY = np.meshgrid(xs, ys)
grids = np.stack([XX.flatten(), YY.flatten()]).T
e_grid = energy_fn(torch.tensor(grids))

# run langevin dynamics
grad_log_p = lambda x: - energy_fn(x)
x0 = torch.randn(n_grid, 2)
l_sample, l_dynamics = sample_langevin(x0, grad_log_p, stepsize, n_steps, intermediate_samples=True)

# plot
fig = plt.figure()
ax = plt.gca()
plt.axis('equal')

point = plt.scatter([],[])
if args.no_arrow:
    arrow = None
else:
    arrow = plt.quiver([0], [0], [1], [1], scale=0.5, scale_units='xy', headwidth=2, headlength=2, alpha=0.3)
plt.tight_layout()

anim = FuncAnimation(fig, update, frames=np.arange(100),
                     init_func=init,
                     interval=200, blit=False)
if args.out is None:
    outfile = f'imgs/{args.energy_function}.gif'
else:
    outfile = f'imgs/{args.out}.gif'
anim.save(outfile, writer='pillow', dpi=80)
print(f'file saved in {outfile}')
