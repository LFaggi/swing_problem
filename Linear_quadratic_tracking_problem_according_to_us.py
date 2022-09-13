import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

n = 20000
T = 250
window_size = 3

a = 1
b = 1
r = 1
q = 1

t_array = torch.linspace(0, T, n)
dt = T / n

threshold = 6


def input_fun(t):
    return 5 * torch.sin(2 * np.pi * 0.01 * t)

class NeuralModel(torch.nn.Module):
    def __init__(self):
        super(NeuralModel, self).__init__()
        self.hidden = 100
        # self.activation = torch.nn.ReLU()
        self.activation = torch.nn.Tanh()
        self.linear_layer1 = torch.nn.Linear(2, self.hidden)
        self.linear_layer2 = torch.nn.Linear(self.hidden, self.hidden)
        self.linear_layer3 = torch.nn.Linear(self.hidden, self.hidden)
        self.linear_layer4 = torch.nn.Linear(self.hidden, 1)
        # self.theta = torch.nn.parameter.Parameter(0*torch.ones(1))

    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.activation(x)
        x = self.linear_layer2(x)
        x = self.activation(x)
        x = self.linear_layer3(x)
        x = self.activation(x)
        x = self.linear_layer4(x)

        # Original proposal
        # x = (self.theta * (x[:,0]-x[:,1]))
        # x = torch.reshape(x,(len(x),1))
        return x

def forward_step(x, p, t_index, model):
    with torch.no_grad():
        input = torch.tensor((x[t_index],input_fun(torch.tensor(t_index * dt)))).unsqueeze(0)
        p[t_index] = model(input)

    # Forward step (Euler discretization) for the state
    x[t_index+1] = x[t_index] + dt * (a * x[t_index]-(b**2/r) * p[t_index])


def backward_learning(x0,p0,t_index,net, window):
    x_train = torch.zeros(window)
    p_train = torch.zeros(window)
    signal_train = torch.zeros(window)

    # Initialization
    p_train[-1] = p0
    x_train[-1] = x0

    # create dataset of optimal solutions going backward
    for window_index in range(window-1):   # TODO da ricontrollare indici segnale input, ricontrollare segni
        x_train[window-window_index-2] = x_train[window-window_index-1] - dt * (a * x_train[window-window_index-1] - (b ** 2 / r) * p_train[window-window_index-1])
        p_train[window - window_index - 2] = p_train[window - window_index-1] - dt * (-q * (x_train[window - window_index - 1] - input_fun(torch.tensor((t_index - window_index) * dt))) - a * p_train[window - window_index - 1])

    for window_index in range(window):
        signal_train[window-window_index-1] = input_fun(torch.tensor((t_index - window_index) * dt))

    # Marco's conditions
    for window_index in range(window-1):
        if abs(x_train[window-window_index-1]) > 100 or abs(p_train[window-window_index-1]) > 100 or abs((x_train[window-window_index-1] - signal_train[window-window_index-1])) < 0.0001:
            x_train = x_train[window-window_index:-1]
            p_train = p_train[window - window_index:-1]
            signal_train = signal_train[window - window_index:-1]
            break

    batch = torch.stack((x_train,signal_train),dim = 1)
    labels = p_train.unsqueeze(dim=1)

    # evaluate model predictions and the corresponding loss
    update_model(net,batch,labels)

def update_model(net, batch, labels):
    outputs = net(batch)

    loss = criterion(outputs, labels)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

if __name__ == "__main__":

    # xf = torch.zeros(n)
    xf = 0.1*torch.ones(n)
    pf = torch.zeros(n)

    theta_for_plot = torch.ones(n)

    model = NeuralModel()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # for par in model.parameters():
    #     print(par.size())

    for t in range(len(t_array)-1):

        backward_learning(xf[t], 0, t, model, window_size)
        forward_step(xf, pf, t, model)

        # for par in model.parameters():
        #     theta_for_plot[t] = par.data

        # Reset if x or p is out of the bounding box
        if abs(xf[t+1])>threshold or abs(pf[t])>threshold:
            xf[t+1] = 0
            pf[t] = 0
        if t % 1000 == 0:
            print(f"Iteration {t}/{n}")

        # forward_step(xf, pf, t, model)
        #
        # if abs(xf[t+1]- input_fun(torch.tensor((t+1)*dt))) > 0.0001 or abs(xf[t+1])<threshold or abs(pf[t])<threshold:  # definire condizione per cui avviene learning
        #     backward_learning(xf[t], 0 , t, model, window_size)
        #
        # for par in model.parameters():
        #     theta[t] = par.data
        #
        # # Reset if x or p is out of the bounding box
        # if abs(xf[t+1])>threshold or abs(pf[t])>threshold:
        #     xf[t+1] = 0
        #     pf[t] = 0

    plt.figure(0)
    plt.plot(t_array,xf, label="State",color="green")
    plt.plot(t_array, pf, label="Costate", color="red")
    plt.plot(t_array,input_fun(t_array), label="Signal", color="cyan")
    plt.ylim((-10,10))

    plt.legend()

    # plt.figure(1)
    # plt.plot(t_array, theta_for_plot, label="Theta", color="orange")
    # plt.ylim((-5,5))
    # plt.legend()

    plt.show()

