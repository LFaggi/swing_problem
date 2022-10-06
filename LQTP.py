import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_bvp

T = 500
n = 20000

a = 1
b = 1
q = 1
r = 0.1

x_0 = 0
p_T = 0

t = np.linspace(0, T, num=n, endpoint=True)
y = np.zeros((2, t.size))


def signal(t):
    arr=[]
    for i in range(len(t)):
        arr.append(5 * math.sin(2 * np.pi * 0.01 * t[i]))  # define the signal to be tracked
    return np.array(arr)

def fun(t, y):
    return np.vstack((a*y[0]-(b**2/r)*y[1], -q*(y[0] - signal(t))-a*y[1]))

def bc(ya, yb):
    return np.array([ya[0]-x_0, yb[1]-p_T])

sol = solve_bvp(fun, bc, t, y, verbose = 2)

print("Initial costate value should be:", sol.sol(0)[1])

t_plot = np.linspace(0, T, 5000)
x_plot = sol.sol(t_plot)[0]
p_plot = sol.sol(t_plot)[1]
sig_plot = signal(t_plot)
plt.plot(t_plot, x_plot, label='State',color="green")
plt.plot(t_plot, p_plot, label='Costate',color="red")
plt.plot(t_plot, sig_plot, label='Signal',color="cyan")
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("t")
plt.xlim(0,T)
plt.ylim(-10,10)
plt.legend()
plt.title("Optimal solution")

plt.show()

