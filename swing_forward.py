import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp

T = 20
n = 10000

# \ddot \theta + lambda_diss \dot \theta + a sin \theta  = u

a = 1 # it is equal to g/l
lambda_diss = 1
lambda_exp = 0.01
r = 1


x1_0 = 1    # initial angle
x2_0 = 0   # initial angular speed

p1_0 = 0
p2_0 = 0

y = np.zeros(4)


def signal(t):
    return np.sin(t) # define the signal to be tracked

def fun(t, y):
    print("Time:> ",t)
    return [y[1], -y[3]/r - lambda_diss * y[1] - a * np.sin(y[0]), -(y[0] - signal(t)) * np.exp(-lambda_exp * (T-t)) + a * y[3] * np.cos(y[0]), -y[2] + lambda_diss * y[3]]

sol = solve_ivp(fun, [0,T], [x1_0, x2_0, p1_0, p2_0], dense_output=True, rtol=10**-10, method="DOP853")


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
plt.ylim(-10.1,10.1)
plt.legend()

plt.show()