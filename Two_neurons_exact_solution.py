import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp

# m_xi is equal to 0 in this script!

# np.random.seed(7)

T = 10
n = 20000

alpha = np.array([1,1])
# theta = [theta11, theta12, theta21, theta22]
theta = np.array([1,1,1,1])
phi = 1

mv = 0.01

xi0 = np.array([0.5,1.])
omega0 = np.random.rand(4)
print(omega0)


p_T = 0

t = np.linspace(0, T, num=n, endpoint=True)

# define the signal to be tracked, it is defined in this strange way to make it work with scipy.bvp routine
def signal(temporal_array):
    arr = []
    for i in range(len(temporal_array)):
        arr.append(0.5 * math.sin(2 * np.pi * 0.01 * temporal_array[i]))
    return np.array(arr)

y = np.zeros((12, t.size))          # initial guess

y[0] = signal(t)

# y = [xi1, xi2, w11, w12, w21, w22, pxi1, pxi2, pw11, pw12, pw21, pw22]
#       0    1    2    3    4    5     6     7    8      9    10    11

def activation_fun(x):
    return np.tanh(x)

def activation_fun_prime(x):
    return 1 - np.tanh(x) ** 2

# y = [xi1, xi2, w11, w12, w21, w22, pxi1, pxi2, pw11, pw12, pw21, pw22]
#       0    1    2    3    4    5     6     7    8      9    10    11

def fun(t, y):
    a1 = theta[0] * y[2] * y[0] + theta[1] * y[3] * y[1]
    a2 = theta[2] * y[4] * y[0] + theta[3] * y[5] * y[1]

    return np.vstack((alpha[0]*(-y[0] + activation_fun(a1)),
                      alpha[1]*(-y[1] + activation_fun(a2)),
                      -y[8]/(phi*mv),
                      -y[9]/(phi*mv),
                      -y[10]/(phi*mv),
                      -y[11]/(phi*mv),
                      -phi * (y[0]-signal(t)) + alpha[0] * y[6] - alpha[0] * y[6] * activation_fun_prime(a1) * theta[0] * y[2] - alpha[1] * y[7] * activation_fun_prime(a2) * theta[2] * y[4],
                      alpha[1] * y[7] - alpha[0] * y[6] * activation_fun_prime(a1) * theta[1] * y[3] - alpha[1] * y[7] * activation_fun_prime(a2) * theta[3] * y[5],
                      - alpha[0] * y[6] * activation_fun_prime(a1) * theta[0] * y[0],
                      - alpha[0] * y[6] * activation_fun_prime(a1) * theta[1] * y[1],
                      - alpha[1] * y[7] * activation_fun_prime(a2) * theta[2] * y[0],
                      - alpha[1] * y[7] * activation_fun_prime(a2) * theta[3] * y[1]))

def bc(ya, yb):
    return np.array([ya[0] - xi0[0],
                     ya[1] - xi0[1],
                     ya[2] - omega0[0],
                     ya[3] - omega0[1],
                     ya[4] - omega0[2],
                     ya[5] - omega0[3],
                     yb[6]-p_T,
                     yb[7]-p_T,
                     yb[8]-p_T,
                     yb[9]-p_T,
                     yb[10]-p_T,
                     yb[11]-p_T])

sol = solve_bvp(fun, bc, t, y, bc_tol=1e-5, verbose=2, max_nodes=500000, tol=1e-9)

t_plot = np.linspace(0, T, 1000)
xi1_plot = sol.sol(t_plot)[0]
xi2_plot = sol.sol(t_plot)[1]
w11_plot = sol.sol(t_plot)[2]
w12_plot = sol.sol(t_plot)[3]
w21_plot = sol.sol(t_plot)[4]
w22_plot = sol.sol(t_plot)[5]
p_xi1_plot = sol.sol(t_plot)[6]
p_xi2_plot = sol.sol(t_plot)[7]
p_w11_plot = sol.sol(t_plot)[8]
p_w12_plot = sol.sol(t_plot)[9]
p_w21_plot = sol.sol(t_plot)[10]
p_w22_plot = sol.sol(t_plot)[11]

sig_plot = signal(t_plot)
plt.plot(t_plot, xi1_plot, label=r'$\xi_1$',color="green")
plt.plot(t_plot, xi2_plot, label=r'$\xi_2$',color="red")
plt.plot(t_plot, w11_plot, label=r'$w$',color="orange")
plt.plot(t_plot, w12_plot, color="orange")
plt.plot(t_plot, w21_plot, color="orange")
plt.plot(t_plot, w22_plot, color="orange")

plt.plot(t_plot, p_xi1_plot, label=r'$p_\xi$',color="blue", linestyle="-.")
plt.plot(t_plot, p_xi2_plot, color="blue", linestyle="-.")
plt.plot(t_plot, p_w11_plot, label=r'$p_w$',color="orange", linestyle="-.")
plt.plot(t_plot, p_w12_plot, color="orange", linestyle="-.")
plt.plot(t_plot, p_w21_plot, color="orange", linestyle="-.")
plt.plot(t_plot, p_w22_plot, color="orange", linestyle="-.")

plt.plot(t_plot, sig_plot, label="signal", color="cyan")

plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("t")
plt.xlim(0,T)
plt.legend()
plt.ylim(-1.1, 4.1)
plt.title("Soluzione backward")


print("--------------------")
print("Initial conditions:")
print("xi_1:> ",xi0[0])
print("xi_2:> ",xi0[1])
print("omega_11:> ",omega0[0])
print("omega_12:> ",omega0[1])
print("omega_21:> ",omega0[2])
print("omega_22:> ",omega0[3])
print("p_xi1:> ",p_xi1_plot[0])
print("p_xi2:> ",p_xi2_plot[0])
print("p_w11:> ",p_w11_plot[0])
print("p_w12:> ",p_w12_plot[0])
print("p_w21:> ",p_w21_plot[0])
print("p_w22:> ",p_w22_plot[0])

plt.show()

# print("Forward....")
#
#
# def right_hand(t, y):  # FOr scipy odeint
#     def input_fun(x):
#         return 1. * np.sin(x) + 0.5
#
#     a1 = theta[0] * y[2] * y[0] + theta[1] * y[3] * y[1]
#     a2 = theta[2] * y[4] * y[0] + theta[3] * y[5] * y[1]
#
#     activations = np.array([a1, a2])
#
#     a = np.array([(alpha[0] * (-y[0] + activation_fun(a1)),
#                    alpha[1] * (-y[1] + activation_fun(a2)),
#                    -y[8] / (phi * mv),
#                    -y[9] / (phi * mv),
#                    -y[10] / (phi * mv),
#                    -y[11] / (phi * mv),
#                    -phi * (y[0] - input_fun(t)) + alpha[0] * y[6] - alpha[0] * y[6] * activation_fun_prime(a1) *
#                    theta[0] * y[2] - alpha[1] * y[7] * activation_fun_prime(a2) * theta[2] * y[4],
#                    alpha[1] * y[7] - alpha[0] * y[6] * activation_fun_prime(a1) * theta[1] * y[3] - alpha[1] * y[
#                        7] * activation_fun_prime(a2) * theta[3] * y[5],
#                    - alpha[0] * y[6] * activation_fun_prime(a1) * theta[0] * y[0],
#                    - alpha[0] * y[6] * activation_fun_prime(a1) * theta[1] * y[1],
#                    - alpha[1] * y[7] * activation_fun_prime(a2) * theta[2] * y[0],
#                    - alpha[1] * y[7] * activation_fun_prime(a2) * theta[3] * y[1])])
#     return np.squeeze(a)
#
# t_eval = np.linspace(0, T, 20000)
#
# y0 = np.array([xi0[0],xi0[1], omega0[0], omega0[1], omega0[2], omega0[3], p_xi1_plot[0], p_xi2_plot[0], p_w11_plot[0], p_w12_plot[0], p_w21_plot[0],p_w22_plot[0]])
#
# sol_forward = solve_ivp(right_hand, (0,20), y0, t_eval=t_eval, rtol=1e-14, atol=1e-14)
# print(sol_forward)
#
# plt.figure()
# plt.plot(sol_forward.t,sol_forward.y[0], label=r'$\xi_1$',color="green")
# plt.plot(sol_forward.t,sol_forward.y[1], label=r'$\xi_2$',color="red")
# plt.plot(sol_forward.t,sol_forward.y[2], label=r'$w$',color="orange")
# plt.plot(sol_forward.t,sol_forward.y[3], color="orange")
# plt.plot(sol_forward.t,sol_forward.y[4], color="orange")
# plt.plot(sol_forward.t,sol_forward.y[5], color="orange")
# plt.plot(sol_forward.t,sol_forward.y[6], label=r'$p_\xi$',color="blue", linestyle="-.")
# plt.plot(sol_forward.t,sol_forward.y[7], color="blue", linestyle="-.")
# plt.plot(sol_forward.t,sol_forward.y[8], label=r'$p_w$',color="orange", linestyle="-.")
# plt.plot(sol_forward.t,sol_forward.y[9], color="orange", linestyle="-.")
# plt.plot(sol_forward.t,sol_forward.y[10], color="orange", linestyle="-.")
# plt.plot(sol_forward.t,sol_forward.y[11], color="orange", linestyle="-.")
#
#
# plt.plot(t_plot, sig_plot, label="signal", color="cyan")
# plt.axhline(y=0, color='black', linestyle='--')
#
# plt.ylim(-1.1, 4.1)
# plt.title("forward")
#
# plt.show()
