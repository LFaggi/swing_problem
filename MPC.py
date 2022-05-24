import numpy as np
import math
import argparse
import os
import matplotlib.pyplot as plt


T = 2
buffer = 0.3
delta_t = 0.005
iterations_per_step = 10

initial_phi = 0
initial_omega = 0.1

g = 9.81
l = 1
m = 1
lambda_diss = 0.1

# initialization
initial_states = [initial_phi, initial_omega]


def signal(t):
    return np.sin(t)

# Initialization

def forward_step(states, costates, t):
    phi_new = states[0] + delta_t * states[1]
    omega_new = states[1] + delta_t * (- np.sin(states[0]) * (g/l) - lambda_diss * states[1] - costates[1]/m**2)
    return phi_new, omega_new

def backward_step(states, costates, t):
    p_phi_old = costates[0] + delta_t * ((states[0]-signal(t))-costates[1] * np.cos(states[0]) * (g/l))
    p_omega_old = costates[1] + delta_t * (-lambda_diss * costates[1] + costates[0])
    return p_phi_old, p_omega_old

def make_step(t, initial_states, final_costates, costate_init = None):
    t_list = np.arange(t, t+buffer, delta_t)

    states_list = [[0,0] for _ in range(len(t_list))]
    states_list[0] = initial_states.copy()
    if costate_init is None:
        costates_list = [[0,0] for _ in range(len(t_list))]
    else:
        costates_list = costate_init
    costates_list[-1] = final_costates.copy()

    for i in range(iterations_per_step):
        states = initial_states
        costates = final_costates

        for k in range(0, len(t_list)-1):
            states[0], states[1] = forward_step(states, costates_list[k], t_list[k])
            states_list[k+1] = [states[0],states[1]]
        for k in reversed(range(1,len(t_list))):
            costates[0], costates[1] = backward_step(states_list[k], costates,  t_list[k])
            costates_list[k-1] = [costates[0],costates[1]]

    return states_list[1], costates_list[1], costates_list

t_list = np.arange(0, T, delta_t)

states_list = [[0,0] for _ in range(len(t_list))]
states_list[0] = initial_states
states = initial_states.copy()
costates_list = [[0,0] for _ in range(len(t_list))]
init_list = None

for k in range(len(t_list)-1):
    print(k)
    states, costates, init_list = make_step(t_list[k], states, [0,0], init_list)
    states_list[k+1] = states
    costates_list[k+1] = costates

states_for_plot0 = []
states_for_plot1 = []

costates_for_plot0 = []
costates_for_plot1 = []

signal_list = []

for k in range(len(states_list)):
    states_for_plot0.append(states_list[k][0])
    states_for_plot1.append(states_list[k][1])
    costates_for_plot0.append(costates_list[k][0])
    costates_for_plot1.append(costates_list[k][1])
    signal_list.append(signal(t_list[k]).copy())


plt.plot(t_list, states_for_plot0, label=r'$\phi$', color="blue")
plt.plot(t_list, states_for_plot1, label=r'$\omega$', color="red")
plt.plot(t_list, costates_for_plot0, label=r'$p_\phi$', color="blue", linestyle = "--")
plt.plot(t_list, costates_for_plot1, label=r'$p_\omega$', color="red", linestyle = "--")

plt.plot(t_list, signal_list, label=r'Signal', color="green")

plt.ylim(-5,5)

plt.title("States and costates")
plt.legend()

plt.show()