import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_bvp

T = 100
n = 10000

# \ddot \theta + lambda_diss \dot \theta + a sin \theta  = u

a = 1 # it is equal to g/l
lambda_diss = 0


x1_0 = 1    # initial angle
x2_0 = 0   # initial angular speed

p1_T = 0
p2_T = 0

t = np.linspace(0, T, num=n, endpoint=True)
y = np.zeros((4, t.size))


def signal(t):
    return np.sin(t)  # define the signal to be tracked

def fun(t, y):
    return np.vstack((y[1], -y[3] - lambda_diss * y[1] - a * np.sin(y[0]), -(y[0] - signal(t)) + a * y[3] * np.cos(y[0]), -y[2] + lambda_diss * y[3]))

def bc(ya, yb):
    return np.array([ya[0]-x1_0, ya[1] - x2_0, yb[2]-p1_T, yb[3]-p2_T])

sol = solve_bvp(fun, bc, t, y,tol=10**-20)

t_plot = np.linspace(0, T, 1000)
x1_plot = sol.sol(t_plot)[0]
x2_plot = sol.sol(t_plot)[1]
p1_plot = sol.sol(t_plot)[2]
p2_plot = sol.sol(t_plot)[3]
sig_plot = signal(t_plot)
plt.plot(t_plot, x1_plot, label=r'$\theta$',color="blue")
plt.plot(t_plot, x2_plot, label=r'$\dot\theta$',color="cyan")
plt.plot(t_plot, p1_plot, label=r'$p_{\theta}$',color="red")
plt.plot(t_plot, p2_plot, label=r'$p_{\dot\theta}$',color="orange")
plt.plot(t_plot, sig_plot, label='Signal',color="green")
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("t")
plt.xlim(0,T)
plt.legend()

plt.show()

print("Initial costate value should be:", sol.sol(0)[2], sol.sol(0)[3])