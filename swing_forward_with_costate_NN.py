import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self,input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

net = Net(input_dim=3)
net.to(torch.double)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

T = 10
n = 1000

# \ddot \theta + lambda_diss \dot \theta + a sin \theta  = u

a = 1 # it is equal to g/l
lambda_diss = 0.1
lambda_exp = 0.


x1_0 = 1.    # initial angle
x2_0 = 0.   # initial angular speed

p1_0 = 0.
p2_0 = 0.

y = np.zeros(4)


def signal(t):
    return np.sin(t) # define the signal to be tracked

def optimize_weights(y,inp,t):
    for i in range(10):
        output = torch.squeeze(net(inp))
        print(output)
        loss = 0.5 * (output[0] - (-(y[0] - signal(t)) * math.exp(-lambda_exp*(T-t)) + a * y[3] * np.cos(y[0])))**2 + 0.5 * (output[1]-(-y[2] + lambda_diss * y[3]))**2
        print(loss)
        loss.backward()
        for par in net.parameters():
            print(par.grad)
        optimizer.step()
        net.zero_grad()
    return


def fun(t, y):
    print("t:> ",t)
    inp = torch.unsqueeze(torch.concat([torch.unsqueeze(torch.tensor(y[0]),dim = 0),torch.unsqueeze(torch.tensor(y[1]),dim = 0),torch.unsqueeze(torch.tensor(signal(t)),dim = 0)],dim=0),dim = 0)
    optimize_weights(y,inp,t)
    output = np.array(torch.squeeze(net(inp)).detach())
    print(output)
    return [y[1], -y[3] - lambda_diss * y[1] - a * np.sin(y[0]), output[0], output[1]]

sol = solve_ivp(fun, [0,T], [x1_0, x2_0, p1_0, p2_0], dense_output=True, rtol=10**-10, method="DOP853")


t_plot = np.linspace(0, T, 100)
signal_plot=[]
for i in range(0,100):
    signal_plot.append(signal(t_plot[i]))

plt.plot(t_plot, sol.sol(t_plot)[0], label=r'$\theta$',color="blue")
plt.plot(t_plot, sol.sol(t_plot)[1], label=r'$\dot\theta$',color="cyan")
plt.plot(t_plot, sol.sol(t_plot)[2], label=r'$p_{\theta}$',color="red")
plt.plot(t_plot, sol.sol(t_plot)[3], label=r'$p_{\dot\theta}$',color="orange")
plt.plot(t_plot, signal_plot, label='Signal',color="green")
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("t")
plt.xlim(0,T)
plt.ylim(-1.1,1.1)
plt.legend()

plt.show()