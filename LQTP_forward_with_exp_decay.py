import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp

T = 10
lambda_decay = 0.5

a = 1
b = 1
q = 1
r = 0.1

x_0 = 1
p_0 = 0.34832914570220636

y = np.zeros(2)


def signal(t):
    return np.array(math.sin(t)) # define the signal to be tracked

def fun(t, y):
    return [a*y[0]-(b**2/r)*y[1], -q*math.exp(-lambda_decay*(T-t))*(y[0] - signal(t))-a*y[1]]


sol = solve_ivp(fun, [0,T], [x_0,p_0], dense_output=True, rtol=10**-10, method="DOP853")


t_plot = np.linspace(0, T, 100)
signal_plot=[]
for i in range(0,100):
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