import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
import torch, torchvision
from torchviz import make_dot

T = 100
lambda_decay = 0.1

a = 1
b = 1
q = 1
r = 0.1

x_0 = 0
p_0 = 0

y = np.zeros(2)
w = torch.tensor(1., requires_grad=True)
x = torch.zeros(1, requires_grad=False)
p = torch.zeros(1, requires_grad=False)

optim = torch.optim.Adam([w], lr=0.001)
n_int = 10


def signal(t):
    return np.array(math.sin(t)) # define the signal to be tracked

def sigma(w):
    return math.tanh(w) # define your own function

def hamiltonian(w,x,p,t):
    h = torch.tensor(0.5 * q * math.exp(-lambda_decay*(T-t)) * (x-signal(t))**2 + 0.5 * r * w**2 + p * (a * x + b * sigma(w)))
    print(h)
    return h

def eval_w_min(w,x,p,t):
    t_tensor = torch.tensor(t,requires_grad=False)
    # TODO trasforma y[0] y[1] in tensori torch x p
    for i in range(n_int):
        loss = hamiltonian(w,x,p,t_tensor)
        make_dot(loss).render("grafico", format="png")
        print(loss)
        print(w)
        loss.backward()
        optim.step()
        w.zero_grad()
        exit()
    return w.detach().cpu().numpy()

def fun(t, y):
    return [a*y[0]+ b * sigma(eval_w_min(w,y[0],y[1],t)), -q*math.exp(-lambda_decay*(T-t))*(y[0] - signal(t))-a*y[1]]


sol = solve_ivp(fun, [0,T], [x_0,p_0], dense_output=True, rtol=10**-10, method="DOP853")

n_samples = 1000
t_plot = np.linspace(0, T, n_samples)
signal_plot=[]
for i in range(0,n_samples):
    signal_plot.append(signal(t_plot[i]))


plt.plot(t_plot, sol.sol(t_plot)[0], label='State',color="blue")
plt.plot(t_plot, sol.sol(t_plot)[1], label='Costate',color="red")
plt.plot(t_plot, signal_plot, label='Signal',color="green")
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("t")
plt.xlim(0,T)
plt.ylim(-1.1,1.1)
plt.legend()

plt.show()