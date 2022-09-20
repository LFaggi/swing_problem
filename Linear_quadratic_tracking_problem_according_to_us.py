import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy

n = 20000
T = 250

epochs = 1
window_size = 5

a = 1
b = 1

q = 1
r = 0.001

t_array = torch.linspace(0, T, n)
dt = T / n

threshold = 10


def input_fun(t):     # Sinusoidal function
    return 5 * torch.sin(2 * np.pi * 0.01 * t)

def mask(t):
    # return 1
    return 0.5*(torch.sign(torch.sin(torch.tensor(2 * np.pi * 0.07 * t))) + 1)

# def input_fun(t):       # Squared-wave function
#     return torch.tensor(float(signal.square(2*np.pi*0.01*t)))

class NeuralModel(torch.nn.Module):
    def __init__(self):
        super(NeuralModel, self).__init__()
        self.hidden = 100
        # self.activation = torch.nn.ReLU()
        self.activation = torch.nn.LeakyReLU()
        # self.activation = torch.nn.Tanh()
        self.linear_layer1 = torch.nn.Linear(2, self.hidden)
        self.linear_layer2 = torch.nn.Linear(self.hidden, self.hidden)
        self.linear_layer3 = torch.nn.Linear(self.hidden, self.hidden)
        self.linear_layer4 = torch.nn.Linear(self.hidden, 1)

        # Original proposal
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
        input = torch.tensor((x[t_index],mask(t_index * dt)*input_fun(torch.tensor(t_index * dt)))).unsqueeze(0)
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
    for window_index in range(window-1):   # TODO da ricontrollare indici segnale input, ricontrollare segni membro destra
        x_train[window-window_index-2] = x_train[window-window_index-1] - dt * (a * x_train[window-window_index-1] - (b ** 2 / r) * p_train[window-window_index-1])
        p_train[window - window_index - 2] = p_train[window - window_index-1] - dt * (-q * (x_train[window - window_index - 1] - mask((t_index - window_index) * dt)*input_fun(torch.tensor((t_index - window_index) * dt))) - a * p_train[window - window_index - 1])

    for window_index in range(window):
        signal_train[window-window_index-1] = mask((t_index - window_index) * dt) * input_fun(torch.tensor((t_index - window_index) * dt))

    # Marco's conditions
    # for window_index in range(window-1):
    #     if abs(x_train[window-window_index-1]) > 50 or abs(p_train[window-window_index-1]) > 50 or abs((x_train[window-window_index-1] - signal_train[window-window_index-1])) < 0.0001:
    #         x_train = x_train[window-window_index:-1]
    #         p_train = p_train[window - window_index:-1]
    #         signal_train = signal_train[window - window_index:-1]
    #         break

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
    xf = 0 * torch.ones(n)
    pf = 0.5 * torch.ones(n) # Indifferente, viene sovrascritto dalla stima del modello nel forward

    theta_for_plot = torch.ones(n)

    model = NeuralModel()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    weights_norm_array = torch.zeros(len(t_array))

    # for par in model.parameters():
    #     print(par.size())

    for t in range(len(t_array)-1):

        for epoch in range(epochs):
            if epoch==0:
                backward_learning(xf[t], 0, t, model, window_size)
                # backward_learning(input_fun(torch.tensor(t*dt)), 0, t, model, window_size)     # proposto da Alesessandro
            else:
                # x_0 = xf[t] * (torch.rand(1) + 0.5) # uniform sampling around xf[t]
                x_0 = xf[t] + 0.5 * torch.randn(1)    # gaussian sampling around xf[t]
                backward_learning(x_0, 0, t, model, window_size)

        # Average Weights' L2 norm for debugging purposes
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        for par in model.parameters():
            if par.requires_grad:
                weights_norm_array[t] += torch.sqrt(torch.sum(par**2))
        weights_norm_array[t] /= total_params

        forward_step(xf, pf, t, model)

        # for par in model.parameters():
        #     theta_for_plot[t] = par.data

        # Reset if x or p is out of the bounding box
        if abs(xf[t+1])>threshold or abs(pf[t])>threshold:
            print("Reset!")
            xf[t+1] = 0
            # pf[t] = 0  # Inutile, tanto nel forward

        if t % 1000 == 0:
            print(f"Iteration {t}/{n}")

        # Last step:
        with torch.no_grad():
            input = torch.tensor((xf[-1], mask(T) * input_fun(torch.tensor(T)))).unsqueeze(0)
            pf[-1] = model(input)

        for par in model.parameters():
            if par.requires_grad:
                weights_norm_array[-1] += torch.sqrt(torch.sum(par**2))
        weights_norm_array[-1] /= total_params


    plt.figure(0)
    plt.plot(t_array,xf, label="State",color="green")
    plt.plot(t_array, pf, label="Costate", color="red")

    signal_array = torch.zeros(len(t_array))
    for t_index in range(len(t_array)):
        signal_array[t_index] = mask(t_index * dt) * input_fun(torch.tensor(t_index * dt))

    plt.plot(t_array, signal_array, label="Signal", color="cyan")
    plt.ylim((-10,10))

    plt.legend()


    # plt.figure(1)
    # plt.plot(t_array, theta_for_plot, label="Theta", color="orange")
    # plt.ylim((-5,5))
    # plt.legend()

    plt.figure(1)
    plt.plot(t_array, weights_norm_array.detach().numpy(), label="Weights norm", color="orange")
    plt.legend()

    plt.show()

