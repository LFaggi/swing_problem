import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot

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

loss_array = []
t_array = []


torch.manual_seed(8)
net = Net(input_dim=3)
net.to(torch.double)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
lr = 1000

T = 10
n = 1000
t_eval = np.linspace(0, T, 1000)

# \ddot \theta + lambda_diss \dot \theta + a sin \theta  = u

a = 1 # it is equal to g/l
lambda_diss = 0.1
lambda_exp = 0.
r = 1.


x1_0 = 0.    # initial angle
x2_0 = 0.   # initial angular speed

p1_0 = 0.
p2_0 = 0.

y = np.zeros(4)


def signal(t):
    return np.sin(t) # define the signal to be tracked

def optimize_weights(y,inp,t):
    optimizer = optim.Adam(net.parameters(), lr=lr) # TODO should the optimizer be re-initialized each step?
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9) # TODO should the optimizer be re-initialized each step?
    for i in range(10):
        output = torch.squeeze(net(inp))
        print(output)
        loss = 0.5 * (output[0] - (-(y[0] - signal(t)) * math.exp(-lambda_exp*(T-t))+ a * y[3] * np.cos(y[0])))**2 \
               + 0.5 * (output[1]-(-y[2] + lambda_diss * y[3]))**2
        # make_dot(loss).render("grafico", format="png") # the computational graph seems ok!
        loss.backward()
        # for par in net.parameters():
        #     print(par.grad)
        optimizer.step()
        net.zero_grad()
    loss_array.append(loss.detach().numpy())
    return


def fun(t, y):
    print("t:> ",t)
    t_array.append(t)
    inp = torch.unsqueeze(torch.concat(
        [torch.tensor([y[0]]), torch.tensor([y[1]]), torch.tensor([signal(t)])], dim=0), dim=0)
    optimize_weights(y,inp,t)
    output = torch.squeeze(net(inp)).detach().numpy()
    print(output)
    return [y[1], -y[3]/r - lambda_diss * y[1] - a * np.sin(y[0]), output[0], output[1]]

sol = solve_ivp(fun, [0,T], [x1_0, x2_0, p1_0, p2_0], dense_output=True, rtol=10**-5, t_eval = t_eval)


t_plot = np.linspace(0, T, 1000)
signal_plot=[]
for i in range(0,1000):
    signal_plot.append(signal(t_plot[i]))

plt.plot(t_plot, sol.sol(t_plot)[0], label=r'$\theta$',color="blue")
plt.plot(t_plot, sol.sol(t_plot)[1], label=r'$\dot\theta$',color="cyan")
plt.plot(t_plot, sol.sol(t_plot)[2], label=r'$p_{\theta}$',color="red")
plt.plot(t_plot, sol.sol(t_plot)[3], label=r'$p_{\dot\theta}$',color="orange")
plt.plot(t_plot, -sol.sol(t_plot)[3]/r, label=r'Control',color="purple")
plt.plot(t_plot, signal_plot, label='Signal',color="green")
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("t")
plt.xlim(0,T)
plt.ylim(-1.1,1.1)
plt.legend()



plt.show()

print(len(loss_array))
print(len(t_array))
# print(t_eval)
plt.plot(t_array,loss_array)

plt.show()
