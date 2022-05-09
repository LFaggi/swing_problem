import numpy as np
import math
import argparse
import os
import time

if os.path.exists(r".\results.png"):
    os.remove("r.\results.png")
# try:
#     os.remove(r"./results.png")
# except OSError:
#     pass

parser = argparse.ArgumentParser()

parser.add_argument("--T", type=float, default=10 ,help="Time Horizon")
parser.add_argument("--delta_t", type=float, default=0.01 ,help="Integration step")
parser.add_argument("--n_neurons", type=int, default=100)

parser.add_argument("--lambda_exp",type=float, default=0.)
parser.add_argument("--lambda_diss",type=float, default=0.)

parser.add_argument("--on_server", type=str, default="no", choices=["no","yes"])

args = parser.parse_args()

import matplotlib
if args.on_server == "yes":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

T = args.T
delta_t = args.delta_t
n_neurons = args.n_neurons

lambda_diss = args.lambda_diss
lambda_exp = args.lambda_exp
g = 9.81
l = 1
m = 1
m_zero = 1
m_theta_n = np.ones((n_neurons,n_neurons))
m_phi =  np.ones(n_neurons)
m_omega =  np.ones(n_neurons)
alpha = np.ones(n_neurons)

def signal(t):
    return np.sin(t)

# Initialization

# State variables
phi = 0
omega = 0
xi = 0.1 * np.random.rand(n_neurons)
theta_n = 0.1 * np.random.rand(n_neurons,n_neurons)
theta_phi = 0.1 * np.random.rand(n_neurons)
theta_omega = 0.1 * np.random.rand(n_neurons)

state_variables = [phi,omega,xi,theta_n,theta_phi,theta_omega]

# Costate variables

p_phi = 0
p_omega = 0
p_xi = np.zeros(n_neurons)
p_theta_n  = np.zeros((n_neurons,n_neurons))
p_theta_phi =   np.zeros(n_neurons)
p_theta_omega =   np.zeros(n_neurons)

costate_variables = [p_phi,p_omega,p_xi,p_theta_n,p_theta_phi,p_theta_omega]


def make_step(states, costates, t):
    new_states = states
    new_costates = costates

    # states update
    new_states[0] = states[0] + states[1] * delta_t
    new_states[1] = states[1] + delta_t * (- (g/l) * math.sin(states[0]) - lambda_diss * states[1] + states[2][0]/m)

    a = np.zeros(n_neurons)
    for i in range(n_neurons):
        for j in range(n_neurons):
            a[i] += states[3][i,j] * states[2][j]
        a[i] += states[4][i] * states[0] + states[5][i] * states[1]

    for i in range(n_neurons):
        new_states[2][i] = states[2][i] + delta_t * (-alpha[i] * states[2][i] + math.tanh(a[i]))
        for j in range(n_neurons):
            new_states[3][i,j] = states[3][i,j] + delta_t*( - math.exp(-lambda_exp * t) * costates[3][i,j] / m_theta_n[i,j])
        new_states[4][i] = states[4][i] + delta_t * (
                    - math.exp(-lambda_exp * t) * costates[4][i] / m_phi[i])
        new_states[5][i] = states[5][i] + delta_t * (
                    - math.exp(-lambda_exp * t) * costates[5][i] / m_omega[i])

    # costates update
    temp_sum1 = 0
    temp_sum2 = 0
    temp_sum3 = np.zeros(n_neurons)
    for i in range(n_neurons):
        temp_sum1 += (1-math.tanh(a[i])**2) * costates[2][i] * states[4][i] # in costate update equation for phi
        temp_sum2 += (1-math.tanh(a[i])**2) * costates[2][i] * states[5][i] # in costate update equation for omega
        for j in range(n_neurons):
            temp_sum3[i] += (1-math.tanh(a[j])) * states[3][j,i] * costates[2][j] # in costate update equation for xi

    new_costates[0] = costates[0] + delta_t * (-math.exp(lambda_exp * t) * (states[0] - signal(t)) + g * costates[1] * math.cos(states[0])/l - temp_sum1)
    new_costates[1] = costates[1] + delta_t * (-temp_sum2 + lambda_diss * costates[1] - costates[0])

    for i in range(n_neurons):
        new_costates[2][i] = costates[2][i] + delta_t * (alpha[i] * costates[2][i]-temp_sum3[i])
        if i == 0:
            new_costates[2][i] += - delta_t * (costates[1]/m + m_zero * states[2][0] * math.exp(lambda_exp * t))
        for j in range(n_neurons):
            new_costates[3][i, j] = costates[3][i,j] + delta_t * (-costates[2][i] * (1 - math.tanh(a[i])**2) * states[2][j])
        new_costates[4][i] = costates[4][i] + delta_t * (-costates[2][i] * (1 - math.tanh(a[i]) ** 2) * states[0])
        new_costates[5][i] = costates[5][i] + delta_t * (-costates[2][i] * (1 - math.tanh(a[i]) ** 2) * states[1])
    return new_states,new_costates

t_array = np.arange(0,T, delta_t)

states_for_plot0 = []
states_for_plot1 = []
states_for_plot2 = []
states_for_plot3 = []
states_for_plot4 = []
states_for_plot5 = []

costates_for_plot0 = []
costates_for_plot1 = []
costates_for_plot2 = []
costates_for_plot3 = []
costates_for_plot4 = []
costates_for_plot5 = []

signal_for_plot = []

for t in t_array:
    states_for_plot0.append(state_variables[0])
    states_for_plot1.append(state_variables[1])
    states_for_plot2.append(state_variables[2])
    states_for_plot3.append(state_variables[3])
    states_for_plot4.append(state_variables[4])
    states_for_plot5.append(state_variables[5])
    costates_for_plot0.append(costate_variables[0])
    costates_for_plot1.append(costate_variables[1])
    costates_for_plot2.append(costate_variables[2])
    costates_for_plot3.append(costate_variables[3])
    costates_for_plot4.append(costate_variables[4])
    costates_for_plot5.append(costate_variables[5])
    signal_for_plot.append(signal(t))
    print("Time:> ",t)
    state_variables, costate_variables = make_step(state_variables, costate_variables, t)

plt.plot(t_array, states_for_plot0, label=r'$\phi$',color="blue")
plt.plot(t_array, states_for_plot1, label=r'$\omega$',color="cyan")
plt.plot(t_array, np.array(states_for_plot2)[:,0], label=r'$\xi_0$')

plt.plot(t_array, costates_for_plot0, label=r'$p_\phi$')
plt.plot(t_array, costates_for_plot1, label=r'$p_\omega$')
plt.plot(t_array, np.array(costates_for_plot2)[:,0], label=r'$p_{\xi_0}$')

plt.plot(t_array, signal_for_plot, label="Signal", color = "green")

plt.ylim(-5,5)
plt.xlim(0,T)
plt.legend()

plt.savefig('./results.png')
if args.on_server == "no":
    plt.show()
plt.close()