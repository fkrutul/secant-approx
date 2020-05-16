import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers
import matplotlib.pyplot as plt
from pathlib import Path
from os import sep

home = str(Path.home())
cur_dir = str(Path.cwd())

layer_number = 8
neuron_number = 5
training_number = 100
value_number = 10000
val_range = [np.pi/2-.17, - (np.pi/2-.17)]

output_file=''.join(['layer_num_', str(layer_number), '-neuron_number_',
                     str(neuron_number), '-training_number_', str(training_number)])

#### Print out Variables ####
x=np.load(''.join([cur_dir, sep, 'evaluation_xvals.npy']))
p=np.load(''.join([cur_dir, sep, 'evaluation_pvals.npy']))

# Initialize the model
model = Sequential()
# build the number of layers iteratively
for i in range(0, layer_number):
    if i == 0:
        model.add(Dense(neuron_number, activation = 'relu', kernel_regularizer = regularizers.l2(0.001), input_shape = (1,)))
    elif i != 0 and i < layer_number - 1:
        model.add(Dense(neuron_number, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)))
    else:
        model.add(Dense(1))
model.compile(optimizer=Adam(),loss='mse')

from keras.utils import plot_model
plot_model(model, to_file=''.join(['model-', 'layer_num_', str(layer_number), '-neuron_number_',
                     str(neuron_number), '-training_number_', str(training_number), '.svg']), 
                    show_shapes=True, show_layer_names=True)

# generate random numbers between the asymptotes
x = np.random.random((value_number,1)) * val_range[0]
x = np.append(x, x * -1) # get negative values too
y = 1/np.cos(x) # reference to approximate

# train model, keep 20% of samples for validation
hist = model.fit(x,y,validation_split = 0.2,
    epochs = training_number,
    batch_size = int(value_number/training_number))

p = np.random.random((value_number,1))*val_range[0] # new data to test on
p = np.append(p, p*-1)

if layer_number == 2 and neuron_number == 20:
    color_real = 'r'
    color_approx = 'k'
elif layer_number == 4 and neuron_number == 10:
    color_real ='r'
    color_approx='g'
elif layer_number == 8 and neuron_number == 5:
    color_real = 'r'
    color_approx = 'b'

#### Print Figures ####
fig1 = plt.figure()
real = plt.scatter(p, 1/np.cos(p), color = color_real)
approx = plt.scatter(p, model.predict(p), color = color_approx)
ypts_real = 1/np.cos(x)
plt.title('Approximation of Secant(x) Using a Neural Network')
plt.xlabel('x (radians)')
plt.ylabel('sec(x)')
plt.legend((real, approx), ('Analytical Value', 'Experimental Approximation'))
plt.show()
fig1.savefig(''.join([cur_dir, sep, 'comparison_', 'layer_num_', str(layer_number), '-neuron_number_',
                    str(neuron_number), '-training_number_', str(training_number), '.svg']))

fig2 = plt.figure()
plt.plot(range(0, len(hist.history['loss'])), hist.history['loss'], color = color_approx)
plt.axis((-2.5, 100, -.1, 3))
plt.title('Neural Network Loss Progression with Training\nApproximation of Secant(x)')
plt.xlabel('Number of Training Iterations')
plt.ylabel('Mean Squared Error')
plt.show()
fig2.savefig(''.join([cur_dir, sep, 'error_', 'layer_num_', str(layer_number), '-neuron_number_',
                     str(neuron_number), '-training_number_', str(training_number), '.svg']))

#### Print out Variables ####
np.savez(''.join([cur_dir, sep, output_file]), p=p, y=y, approx=model.predict(p), error=hist.history['loss'],
         layer_number=layer_number, neuron_number=neuron_number, 
         training_number=training_number, value_number=value_number, batch=int(value_number/training_number),
         ypts_real=ypts_real, history=hist.history)

np.save(''.join([cur_dir, sep, 'evaluation_xvals']), x)
np.save(''.join([cur_dir, sep, 'evaluation_pvals']), p)
