import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

import argparse

parser = argparse.ArgumentParser(description='MPC neural approach for the LQTP')

parser.add_argument('--T', type=float, default=100.)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--window_mpc', type=float, default=1)
parser.add_argument('--window_training', type=float, default=0.5)
parser.add_argument('--a', type=float, default=1.)
parser.add_argument('--b', type=float, default=1.)
parser.add_argument('--r', type=float, default=1.)
parser.add_argument('--q', type=float, default=1.)


args = parser.parse_args()



T = args.T
dt = args.dt

window_mpc = args.window_mpc          # in seconds
window_training = args.window_training       # in seconds

a = args.a
b = args.b
r = args.r
q = args.q


def input_fun(t):
    return 5 * np.sin(2 * np.pi * 0.01 * t)


class NeuralModel(torch.nn.Module):
    def __init__(self):
        super(NeuralModel, self).__init__()
        self.hidden = 5
        # self.activation = torch.nn.ReLU()
        self.activation = torch.nn.Tanh()
        self.linear_layer1 = torch.nn.Linear(2, self.hidden)
        self.linear_layer2 = torch.nn.Linear(self.hidden, self.hidden)
        self.linear_layer3 = torch.nn.Linear(self.hidden, 1)

    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.activation(x)
        x = self.linear_layer2(x)
        x = self.activation(x)
        x = self.linear_layer3(x)
        return x


def forward_step(x, p, t_ind, neural_model):
    with torch.no_grad():
        state = x[t_index].unsqueeze(0)
        sig = torch.tensor(input_fun(t_ind * dt)).to(torch.float32).unsqueeze(0)
        input_batch = torch.concat((state, sig)).unsqueeze(0)

        p[t_index] = neural_model(input_batch)   # costate prediction

    # Forward step (Euler discretization) for the state
    x[t_ind+1] = x[t_ind] + dt * (a * x[t_ind]-(b**2/r) * p[t_ind])


#----------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------- MPC FUNCTION (FOR SUPERVISION) ----------------------------------


def fun(t, y):
    return np.vstack((a*y[0]-(b**2/r)*y[1], -q*(y[0] - input_fun(t))-a*y[1]))


def MPC(x0, pT, window, t):
    def bc(ya, yb):
        return np.array([ya[0] - x0, yb[1] - pT])

    temporal_array = np.linspace(t, t + window, num=int(window//dt), endpoint=True)
    y = np.zeros((2, temporal_array.size))
    sol = solve_bvp(fun, bc, temporal_array, y, max_nodes=10000, verbose=0)

    if sol.status != 0:
        print("Some problems with MPC occurred", sol.status)

    xf_next = sol.sol(temporal_array)[0][1]
    pf_next = sol.sol(temporal_array)[1][1]
    sig_next = input_fun(temporal_array[1])

    return xf_next, pf_next, sig_next


def generate_dataset(x0, pT, t_ind, w_training, w_mpc):
    n_point = int(w_training//dt)
    x_train = np.zeros(n_point)
    p_train = np.zeros(n_point)
    signal_train = np.zeros(n_point)
    x_train[0] = x0

    # create the dataset through MPC
    for i in range(n_point - 1):
        x_train[i+1], p_train[i+1], signal_train[i+1] = MPC(x_train[i], pT, w_mpc, (t_ind+i) * dt)

    x_train = torch.from_numpy(x_train[1:])
    p_train = torch.from_numpy(p_train[1:])
    signal_train = torch.from_numpy(signal_train[1:])

    batch = torch.stack((x_train,signal_train), dim=1).to(torch.float32)
    labels = p_train.unsqueeze(dim=1).to(torch.float32)

    return (batch, labels)


def update_model(net, data):
    batch = data[0]
    labels = data[1]
    outputs = net(batch)

    loss = criterion(outputs, labels)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()


if __name__ == "__main__":

    n = int(T//dt)
    t_array = np.linspace(0, T, n)

    xf = 0.1*torch.ones(n)
    pf = torch.zeros(n)

    model = NeuralModel()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    weights_norm_array = torch.zeros(len(t_array)-1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for t_index in range(int(3*n//4)-1):

        # Average Weights' L2 norm
        for par in model.parameters():
            if par.requires_grad:
                weights_norm_array[t_index] += torch.sqrt(torch.sum(par ** 2))
        weights_norm_array[t_index] /= total_params

        dataset = generate_dataset(xf[t_index], 0, t_index, window_training, window_mpc)
        update_model(model, dataset)
        forward_step(xf, pf, t_index, model)

        print(t_index, " out of ", n)

    for t_index in range(int(3*n//4) - 1, n-1):

        # Average Weights' L2 norm
        for par in model.parameters():
            if par.requires_grad:
                weights_norm_array[t_index] += torch.sqrt(torch.sum(par ** 2))
        weights_norm_array[t_index] /= total_params

        # dataset = generate_dataset(xf[t_index], 0, t_index, model, window_training, window_mpc)
        # update_model(model, dataset)
        forward_step(xf, pf, t_index, model)

        print(t_index, " out of ", n)

    plt.figure(0)
    plt.plot(t_array,xf, label="State",color="green")
    plt.plot(t_array, pf, label="Costate", color="red")
    plt.plot(t_array,input_fun(t_array), label="Signal", color="cyan")
    plt.axhline(y=0, color='black', linestyle='--')
    plt.ylim((-10,10))
    plt.legend()
    plt.savefig("state_costate.pdf", dpi=500)

    plt.figure(1)
    plt.plot(t_array[:-1], weights_norm_array.detach().numpy(), label="Weights norm", color="orange")
    plt.legend()
    plt.savefig("weights_norm.pdf", dpi=500)

    plt.show()

