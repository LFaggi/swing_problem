import numpy as np
import math
import argparse
import os

# Exponent exp(-lambda * (T-t))
# Reset of costates if |costate| > toll; states evolve continuously in time instead
# Control given by the average of n_c neurons
# weights constrained inside a box

np.random.seed(7)

try:
    os.remove(r"./images/results.png")
    os.remove(r"./images/hamiltonian.png")
    os.remove('./images/costates.png')
    os.remove('./images/activations.png')
except OSError:
    pass

parser = argparse.ArgumentParser()

parser.add_argument("--T", type=float, default=30., help="Time Horizon")
parser.add_argument("--delta_t", type=float, default=0.01, help="Integration step")
parser.add_argument("--n_neurons", type=int, default=20)
parser.add_argument("--n_c", type=int, default=10)
parser.add_argument("--box_width", type=float, default=10.)

parser.add_argument("--alpha", type=float, default=1.)
parser.add_argument("--lambda_exp", type=float, default=0.1)
parser.add_argument("--lambda_diss", type=float, default=1.)
parser.add_argument("--m", type=float, default=1.)
parser.add_argument("--m_zero", type=float, default=1.)
parser.add_argument("--m_theta_n", type=float, default=20.)
parser.add_argument("--m_phi", type=float, default=20.)
parser.add_argument("--m_omega", type=float, default=20.)
parser.add_argument("--tol", type=float, default=2.)
parser.add_argument("--plot_range", type=float, default=2.5)

parser.add_argument("--on_server", type=str, default="no", choices=["no", "yes"])

args = parser.parse_args()

import matplotlib

if args.on_server == "yes":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

T = args.T
delta_t = args.delta_t
n_neurons = args.n_neurons
n_c = args.n_c
box_width = args.box_width

lambda_diss = args.lambda_diss
lambda_exp = args.lambda_exp
g = 9.81
l = 1
m = args.m
m_zero = args.m_zero
m_theta_n = args.m_theta_n * np.ones((n_neurons, n_neurons))
m_phi = args.m_phi * np.ones(n_neurons)
m_omega = args.m_omega * np.ones(n_neurons)
alpha = args.alpha * np.ones(n_neurons)

tol = args.tol

def signal(t):
    # return np.sin(t * 0.01)
    return -0.1 * np.ones(1)

# Initialization

# State variables
phi = 0.
omega = 0.01
xi = 0.01 * np.random.rand(n_neurons)
theta_n = 0.1 * np.random.rand(n_neurons, n_neurons)
theta_phi = 0.1 * np.random.rand(n_neurons)
theta_omega = 0.1 * np.random.rand(n_neurons)

state_variables = [phi, omega, xi, theta_n, theta_phi, theta_omega]

# Costate variables

p_phi = 0.
p_omega = 0.
p_xi = np.zeros(n_neurons)
p_theta_n = np.zeros((n_neurons, n_neurons))
p_theta_phi = np.zeros(n_neurons)
p_theta_omega = np.zeros(n_neurons)

costate_variables = [p_phi, p_omega, p_xi, p_theta_n, p_theta_phi, p_theta_omega]

n_reset = 0
reset_list = []

# initial activations:
activations = np.zeros(n_neurons)
for i in range(n_neurons):
    for j in range(n_neurons):
        activations[i] += state_variables[3][i, j] * state_variables[2][j]
    activations[i] += state_variables[4][i] * state_variables[0] + state_variables[5][i] * state_variables[1]


def make_step(states, costates, t):
    reset = False

    # def reset_fun(i, new_states, new_costates):
    #     new_costates[2][i] = 0
    #     for j in range(n_neurons):
    #         new_costates[3][i,j]=0
    #     new_costates[4][i] = 0
    #     new_costates[5][i] = 0
    #
    #
    #     new_states[2][i] = 0
    #     for j in range(n_neurons):
    #         new_states[3][j, i] = 0.1 * np.random.rand(1)
    #         # new_states[3][j, i] = 0.
    #
    #     new_states[4][i] = 0.1 * np.random.rand(1)
    #     new_states[5][i] = 0.1 * np.random.rand(1)
    #     # new_states[4][i] = 0.
    #     # new_states[5][i] = 0.
    #     return

    new_states = [np.zeros(1), np.zeros(1), np.zeros(xi.shape), np.zeros(theta_n.shape),
                  np.zeros(theta_phi.shape), np.zeros(theta_omega.shape)]
    new_costates = [np.zeros(1), np.zeros(1), np.zeros(xi.shape), np.zeros(theta_n.shape),
                  np.zeros(theta_phi.shape), np.zeros(theta_omega.shape)]

    # new_states = copy.deepcopy(states)     ##liste di array numpy, per copiare la dimensione si inizializzano così
    # new_costates = copy.deepcopy(costates)

    # control as average of first n_c neurons
    control = 0
    for i in range(n_c):
        control += states[2][i]
    control /= n_c

    # a[i] = activation of \xi_i neuron
    a = np.zeros(n_neurons)
    for i in range(n_neurons):
        for j in range(n_neurons):
            a[i] += states[3][i, j] * states[2][j]
        a[i] += states[4][i] * states[0] + states[5][i] * states[1]

    # states update
    new_states[0] = states[0] + states[1] * delta_t
    new_states[1] = states[1] + delta_t * (- (g / l) * math.sin(states[0]) - lambda_diss * states[1] + control / m)

    for i in range(n_neurons):
        new_states[2][i] = states[2][i] + delta_t * (-alpha[i] * states[2][i] + alpha[i] * math.tanh(a[i]))
        for j in range(n_neurons):
            new_states[3][i, j] = states[3][i, j] + delta_t * (
                        - math.exp(-lambda_exp * (t-T)) * costates[3][i, j] / m_theta_n[i, j])

            if new_states[3][i, j] > box_width:
                new_states[3][i, j] = states[3][i, j]

        new_states[4][i] = states[4][i] + delta_t * (
                - math.exp(-lambda_exp * (t-T)) * costates[4][i] / m_phi[i])

        if new_states[4][i] >  box_width:
            new_states[4][i] = states[4][i]

        new_states[5][i] = states[5][i] + delta_t * (
                - math.exp(-lambda_exp * (t-T)) * costates[5][i] / m_omega[i])

        if new_states[5][i] >  box_width:
            new_states[5][i] = states[5][i]

    # costates update
    temp_sum1 = 0
    temp_sum2 = 0
    temp_sum3 = np.zeros(n_neurons)
    for i in range(n_neurons):
        temp_sum1 += alpha[i] * (1 - math.tanh(a[i]) ** 2) * costates[2][i] * states[4][i]             # in costate update equation for phi
        temp_sum2 += alpha[i] * (1 - math.tanh(a[i]) ** 2) * costates[2][i] * states[5][i]               # in costate update equation for omega
        for j in range(n_neurons):
            temp_sum3[i] += alpha[j] * (1 - math.tanh(a[j]) ** 2) * states[3][j, i] * costates[2][j]   # in costate update equation for xi

    new_costates[0] = costates[0] + delta_t * (
                -math.exp(lambda_exp * (t-T)) * (states[0] - signal(t)) + g * costates[1] * math.cos(
            states[0]) / l - temp_sum1)

    if abs(new_costates[0]) > tol:
        reset = True
        new_costates[0] = 0.5 * np.random.rand(1)

    new_costates[1] = costates[1] + delta_t * (-temp_sum2 + lambda_diss * costates[1] - costates[0])

    if abs(new_costates[1]) > tol:
        reset = True
        new_costates[1] = 0.5 * np.random.rand(1)

    for i in range(n_neurons):
        new_costates[2][i] = costates[2][i] + delta_t * (alpha[i] * costates[2][i] - temp_sum3[i])
        if i < n_c:
            new_costates[2][i] += - delta_t * (costates[1] / (m * n_c) + m_zero * control * math.exp(lambda_exp * (t-T)) * (1 / n_c))

        if abs(new_costates[2][i]) > tol:
            reset = True
            new_costates[2][i] = 0.5 * np.random.rand(1)

        for j in range(n_neurons):
            new_costates[3][i, j] = costates[3][i, j] + delta_t * (
                        - alpha[i] * costates[2][i] * (1 - math.tanh(a[i]) ** 2) * states[2][j])
            if abs(new_costates[3][i, j]) > tol:
                reset = True
                new_costates[3][i, j] = 0.5 * np.random.rand(1)

        new_costates[4][i] = costates[4][i] + delta_t * (- alpha[i] * costates[2][i] * (1 - math.tanh(a[i]) ** 2) * states[0])
        if abs(new_costates[4][i]) > tol:
            reset = True
            new_costates[4][i] = 0.5 * np.random.rand(1)

        new_costates[5][i] = costates[5][i] + delta_t * (- alpha[i] * costates[2][i] * (1 - math.tanh(a[i]) ** 2) * states[1])
        if abs(new_costates[5][i]) > tol:
            reset = True
            new_costates[5][i] = 0.5 * np.random.rand(1)

    return new_states.copy(), new_costates.copy(), a, reset


def compute_H(states, costates, t):
    # control as average of first n_c neurons
    control = 0
    for i in range(n_c):
        control += states[2][i]
    control /= n_c

    a = np.zeros(n_neurons)
    temp = np.zeros(n_neurons)
    for i in range(n_neurons):
        for j in range(n_neurons):
            a[i] += states[3][i, j] * states[2][j]
            temp[i] += costates[3][i, j] ** 2 / m_theta_n[i, j]
        a[i] += states[4][i] * states[0] + states[5][i] * states[1]

    H = 0.5 * np.exp(lambda_exp * (t - T)) * ((states[0] - signal(t)) ** 2 + m_zero * control ** 2) + \
        + costates[0] * states[1] + costates[1] * (
                    -g * math.sin(states[0]) / l - lambda_diss * states[1] + control / m)
    for i in range(n_neurons):
        H += -0.5 * np.exp(-lambda_exp * (t-T)) * (
                    temp[i] + costates[4][i] ** 2 / m_phi[i] + costates[5][i] ** 2 / m_omega[i]) + \
             costates[2][i] * (-alpha[i] * states[2][i] + alpha[i] * math.tanh(a[i]))
    return H

t_array = np.arange(0, T, delta_t)

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

activations_for_plot = []

hamiltonian_for_plot = []

signal_for_plot = []

for t in t_array:
    states_for_plot0.append(state_variables[0])
    states_for_plot1.append(state_variables[1])
    states_for_plot2.append(state_variables[2].copy())
    states_for_plot3.append(state_variables[3].copy())
    states_for_plot4.append(state_variables[4].copy())
    states_for_plot5.append(state_variables[5].copy())
    costates_for_plot0.append(costate_variables[0])
    costates_for_plot1.append(costate_variables[1])
    costates_for_plot2.append(costate_variables[2].copy())
    costates_for_plot3.append(costate_variables[3].copy())
    costates_for_plot4.append(costate_variables[4].copy())
    costates_for_plot5.append(costate_variables[5].copy())
    activations_for_plot.append(activations.copy())
    signal_for_plot.append(signal(t))

    hamiltonian_for_plot.append(compute_H(state_variables, costate_variables, t))

    print("Time:> ", t)

    state_variables, costate_variables, activations, reset = make_step(state_variables, costate_variables, t)

    if reset:
        reset_list.append(t)
        n_reset += 1

print(f"# resets: %i" % n_reset)

plt.figure(1)
plt.title("States and costates")
plt.plot(t_array, states_for_plot0, label=r'$\phi$', color="blue")
plt.plot(t_array, states_for_plot1, label=r'$\omega$', color="red")
plt.plot(t_array, np.array(states_for_plot2)[:, 0], label=r'$\xi_0$', color="orange")

plt.plot(t_array, costates_for_plot0, label=r'$p_\phi$', color="blue", linestyle="-.")
plt.plot(t_array, costates_for_plot1, label=r'$p_\omega$', color="red", linestyle="-.")
plt.plot(t_array, np.array(costates_for_plot2)[:, 0], label=r'$p_{\xi_0}$', color="orange", linestyle="-.")

plt.plot(t_array, signal_for_plot, label="Signal", color="green")

plt.ylim(-args.plot_range, args.plot_range)
plt.xlim(0, T)
plt.legend()

plt.savefig('./images/results.png')

plt.figure(2)
plt.title("Hamiltonian")
plt.plot(t_array, hamiltonian_for_plot, label='$H$', color="red")
plt.savefig('./images/hamiltonian.png')

plt.figure(3)
plt.plot(t_array, [tol for _ in range(len(t_array))], color="black", linestyle=":")
plt.plot(t_array, [-tol for _ in range(len(t_array))], color="black", linestyle=":")

plt.plot(t_array, np.array(costates_for_plot2)[:, 0], label=r"$p_{\xi_0}$")
for i in range(1, n_neurons):
    plt.plot(t_array, np.array(costates_for_plot2)[:, i])

for k in range(len(reset_list)):
    plt.plot(reset_list[k], 0, marker="+", color="black", markersize=1)

plt.ylim(-args.plot_range, args.plot_range)
plt.legend()
plt.savefig('./images/costates.png')

plt.figure(4)
plt.title("Activations")

plt.plot(t_array, np.array(activations_for_plot)[:, 0],label=r"$a_0$")
for i in range(1, n_neurons):
    plt.plot(t_array, np.array(activations_for_plot)[:, i])
plt.legend()
plt.savefig('./images/activations.png')

if args.on_server == "no":
    plt.show()
plt.close()