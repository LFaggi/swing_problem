import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

a = 1
b = 1
q = 1
r = 0.1

x_0 = 0
p_T = 0

T = 60
dt = 0.001
window_size = 100

# def input_fun(t):
#     return 5 * torch.sin(2 * np.pi * 0.01 * t)

def signal(t): #t must be passed as an ARRAY!
    return 5 * np.sin(2 * np.pi * 0.01 * t)

def fun(t, y):
    return np.vstack((a*y[0]-(b**2/r)*y[1], -q*(y[0] - signal(t))-a*y[1]))

def bc(ya, yb, x_0, p_T):
    return np.array([ya[0]-x_0, yb[1]-p_T])


def MPC(x0, window, pT, t):
    def bc(ya, yb):
        return np.array([ya[0] - x0, yb[1] - pT])
    t_array = np.linspace(t, t + window, num=int(window//dt), endpoint=True)
    y = np.zeros((2, t_array.size))
    sol = solve_bvp(fun, bc, t_array, y, max_nodes=10000, verbose=0)
    xf = sol.sol(t_array)[0][1]
    pf = sol.sol(t_array)[1][1]
    if sol.status != 0:
        print("Porca puttana!", sol.status)
    return xf, pf


#----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------- MAIN PROGRAM ---------------------------------------------------

if __name__ == "__main__":

    t_plot = np.linspace(0, T, num= int(T // dt), endpoint=True)

    xf = 0. * np.ones(len(t_plot))
    pf = 0. * np.ones(len(t_plot))

    for t in range(len(t_plot)-1):
        print("Time: ", t*dt)
        print("Xf: ", xf[t])
        xf[t+1], pf[t+1] = MPC(xf[t], window_size*dt, 0, t*dt)


    plt.figure(0)
    plt.plot(t_plot,xf, label="State",color="green")
    plt.plot(t_plot, pf, label="Costate", color="red")
    plt.plot(t_plot, signal(t_plot), label="Signal", color="cyan")
    plt.axhline(y=0, color='black', linestyle='--')

    plt.legend()

    plt.show()