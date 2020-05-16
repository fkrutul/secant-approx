import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from os import sep

home = str(Path.home())
cur_dir = str(Path.cwd())
layer_number = [2, 4, 8]
neuron_number = [20, 10, 5]
training_number = 100
value_number = 10000
val_range = [np.pi/2-.17, -(np.pi/2-.17)]

#### Print out Variables ####
x = np.load(''.join([cur_dir, sep, 'evaluation_xvals.npy']))
p = np.load(''.join([cur_dir, sep, 'evaluation_pvals.npy']))

approx = []
real = []

fig1 = plt.figure()
color = ['k','g','b']
for i in range(0, len(layer_number)):
    output_file = ''.join(['layer_num_', str(layer_number[i]), '-neuron_number_',
                     str(neuron_number[i]), '-training_number_', str(training_number)])
    data = np.load(''.join([cur_dir, sep, output_file, '.npz']))
    p = data.f.p
    predict_approx=data.f.approx
    batch = data.f.batch
    error = data.f.error
    y = data.f.y
    ypts_real = data.f.ypts_real
    hist = data.f.history
    
    #### Print Figures ####
    if i == 1:
        real.append(plt.scatter(p, 1/np.cos(p), color = 'r', s = 4))
    approx.append(plt.scatter(p, predict_approx, color = color[i], s = 4))
    plt.title('Approximation of Secant(x) Using a Neural Network')
    plt.xlabel('x (radians)')
    plt.ylabel('sec(x)')
plt.legend((real[0], approx[0], approx[1], approx[2]), ('Analytical Value', 'Approx 2:20', 'Approx 4:10', 'Approx 8:5'))
fig1.savefig(''.join([cur_dir, sep, 'comparison_', 'layer_num_', str(layer_number), '-neuron_number_',
                         str(neuron_number), '-training_number_', str(training_number), '_multiple.svg']))

loss = []
fig2 = plt.figure()
for i in range(0, 3):
    output_file = ''.join(['layer_num_', str(layer_number[i]), '-neuron_number_',
                     str(neuron_number[i]), '-training_number_', str(training_number)])
    data = np.load(''.join([cur_dir, sep, output_file, '.npz']))
    p = data.f.p
    predict_approx = data.f.approx
    batch = data.f.batch
    error = data.f.error
    y = data.f.y
    ypts_real = data.f.ypts_real
    hist = data.f.history
    loss.append(plt.plot(range(0, len(error)), error))

plt.axis((-2.5, 100, -.1, 3))
plt.title('Neural Network Loss Progression with Training\nApproximation of Secant(x)')
plt.xlabel('Number of Training Iterations')
plt.ylabel('Mean Squared Error')
plt.legend((loss[0][0], loss[1][0], loss[2][0]), ('Loss 2:20', 'Loss 4:10', 'Loss 8:5'))
fig2.savefig(''.join([cur_dir, sep, 'error_', 'layer_num_', str(layer_number), '-neuron_number_',
                             str(neuron_number), '-training_number_', str(training_number), '_multiple.svg']))
