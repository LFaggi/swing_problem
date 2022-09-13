import numpy as np
import torch

n = 1000
T = 10
window = 5
alpha = torch.tensor([1,1])
phi = 1
mv = 1

t_array = np.linspace(0, T, n, endpoint=True)
dt = T/n

def input_fun(t):
    return 1. * np.sin(t) + 0.5

# y = [xi1, xi2, w11, w12, w21, w22, pxi1, pxi2, pw11, pw12, pw21, pw22]
#       0    1    2    3    4    5     6     7    8      9    10    11

yf = torch.zeros((12,len(t_array)))
yb = torch.zeros((12,len(t_array)))

# Initialization
yf[0,0] = 1
yf[1,0] = 1
w0 = 0.1 * torch.rand(4)
yf[2,0] = w0[0]
yf[3,0] = w0[1]
yf[4,0] = w0[2]
yf[5,0] = w0[3]

def activation_fun(x):
    return np.tanh(x)

def activation_fun_prime(x):
    return 1 - np.tanh(x) ** 2



class NeuralModel(torch.nn.Module):

    def __init__(self):
        super(NeuralModel, self).__init__()
        self.hidden = 100
        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.torch.nn.Linear(1, self.hidden)(x)
        x = self.activation(x)
        x = self.torch.nn.Linear(self.hidden, self.hidden)(x)
        x = self.activation(x)
        x = self.torch.nn.Linear(self.hidden, 1)(x)
        return x



def compute_activations(x):
    a1 = x[2] * x[0] + x[3] * x[1]
    a2 = x[4] * x[0] + x[5] * x[1]
    return torch.tensor([a1,a2])

def forward_step(y,t_index,models):
    activations = compute_activations(y[:, t_index])
    with torch.no_grad:
        y[6,t_index] = models[0](y[0,t_index])
        y[7, t_index] = models[1](y[1, t_index])
        y[8, t_index] = models[2](y[2, t_index])
        y[9, t_index] = models[3](y[3, t_index])
        y[10, t_index] = models[4](y[4, t_index])
        y[11, t_index] = models[5](y[5, t_index])

    y[0, t_index+1] = y[0, t_index] + dt * alpha[0] * (-y[0,t_index] + activation_fun(activations[0]))
    y[1, t_index + 1] = y[1, t_index] + dt * alpha[1] * (-y[1,t_index] + activation_fun(activations[1]))

    y[2, t_index + 1] = y[2, t_index] - dt * y[8, t_index]/(phi*mv)
    y[3, t_index + 1] = y[3, t_index] - dt * y[9, t_index]/(phi*mv)
    y[4, t_index + 1] = y[4, t_index] - dt * y[10, t_index]/(phi*mv)
    y[5, t_index + 1] = y[5, t_index] - dt * y[11, t_index]/(phi*mv)

def backward_learning(y, t_index, window): # backward Euler
    for k in range(1,len(window)):

        activations = compute_activations(y[:, t_index + window - k + 1])
        y[0, t_index+window-k] = y[0, t_index+window - k + 1]  - dt * alpha[0] * (-y[0,t_index] + activation_fun(activations[0]))
        y[1, t_index + window - k] = y[1, t_index + window - k + 1] - dt * alpha[1] * (-y[1, t_index] + activation_fun(activations[1]))

        y[2, t_index + window - k] = y[2, t_index + window - k + 1] - dt * y[8,  t_index + window - k + 1]/(phi*mv)
        y[3, t_index + window - k] = y[3, t_index + window - k + 1] - dt * y[9, t_index + window - k + 1] / (phi * mv)
        y[4, t_index + window - k] = y[4, t_index + window - k + 1] - dt * y[10, t_index + window - k + 1] / (phi * mv)
        y[5, t_index + window - k] = y[5, t_index + window - k + 1] - dt * y[11, t_index + window - k + 1] / (phi * mv)

        y[6, t_index + window - k] = y[6, t_index + window - k + 1] - dt * (-phi * (y[0,  t_index + window - k + 1]-input_fun(t_array[t_index + window - k + 1])) + alpha[0] * y[6, t_index + window - k + 1] - alpha[0] * y[6, t_index + window - k + 1] * activation_fun_prime(activations[1]) * y[2,  t_index + window - k + 1] - alpha[1] * y[7,  t_index + window - k + 1] * activation_fun_prime(activations[2]) * y[4, t_index + window - k + 1])
        y[7, t_index + window - k] = y[7, t_index + window - k + 1] - dt * (alpha[1] * y[7,t_index + window - k + 1] - alpha[0] * y[6,t_index + window - k + 1] * activation_fun_prime(activations[0]) * y[3, t_index + window - k + 1] - alpha[1] * y[7] * activation_fun_prime(activations[1]) * y[5, t_index + window - k + 1])
        y[8, t_index + window - k] = y[8, t_index + window - k + 1] - dt * (- alpha[0] * y[6,t_index + window - k + 1] * activation_fun_prime(activations[0]) * y[0, t_index + window - k + 1])
        y[8, t_index + window - k] = y[8, t_index + window - k + 1] - dt * (