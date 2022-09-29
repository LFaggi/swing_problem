import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Tracking di segnale sinusoidale con predizione dei costati p_xi da rete neurale.
# Fase backward di apprendimento della relazione stato/costato seguita da step forawrd.
# Finestra in avanti al posto che all'indietro e più epoche.

#Definition of the time horizon and temporal nodes
n = 20000
T = 250
window_size = 5
epochs = 5

t_array = torch.linspace(0, T, n)
dt = T / n

threshold = 100

n_neurons = 2

#Definition of hyperparameters

a_1 = 1
a_2 = 0
theta = torch.ones(n_neurons, n_neurons)
alpha = torch.ones(n_neurons)
phi = 1.
m_omega = 1. * torch.ones(n_neurons, n_neurons)   # TODO da sperimentare al variare di m_omega
m_xi = 0. * torch.ones(n_neurons)

torch.manual_seed(77)
#----------------------------------------------------------------------------------------------------------------------

# Compute activations
def compute_activations(xi, omega, theta, t_index):
    a = torch.zeros(n_neurons)
    for i in range(n_neurons):
        for j in range(n_neurons):
            a[i] += theta[i][j] * omega[t_index][i][j] * xi[t_index][j]
    return a

#Activation function and derivative of the neurons

#Linear activation
# def activation_fun(a):
#     return a
#
# def activation_fun_prime(a):
#     return torch.tensor(1)


# # Tanh activation
def activation_fun(a):
    return torch.tanh(a)

def activation_fun_prime(a):
    return 1 - torch.tanh(a)**2


#ReLU activation
# def activation_fun(a):
#     if a >= 0:
#         return a.clone().detach()
#     elif a<0:
#         return torch.tensor(0)
#
# def activation_fun_prime(a):
#     if a >=0:
#         return torch.tensor(1)
#     elif a<0:
#         return torch.tensor(0)

#Definition of the potential U    # TODO per uniformità con le altre funzioni passargli il t_index
def potential(xi,u):
    return 0.5 * a_1 * (xi[0] - u)**2 + 0.5 * a_2 * (xi[-1] - xi[0])**2


#Prime derivative of the potential U
def potential_prime(xi,u,i):
    if i == 0:
        return a_1 * (xi[0] - u) - a_2 * (xi[-1] - xi[0])
    elif i == n_neurons-1:
        return a_2 * (xi[-1] - xi[0])
    else:
        return 0


#Definition of the target signal
def input_fun(t):
    return 0.5 * torch.sin(2 * np.pi * 0.01 * t)


#---------------------------------------------- NEURAL MODEL FOR PREDICTING P_XI --------------------------------------

#----------------------------------------------------------------------------------------------------------------------

class NeuralModel(torch.nn.Module):
    def __init__(self):
        super(NeuralModel, self).__init__()
        self.hidden = 10
        self.n_neurons = n_neurons
        self.activation = torch.nn.ReLU()
        # self.activation = torch.nn.Tanh()
        self.linear_layer1 = torch.nn.Linear(self.n_neurons + 1, self.hidden)
        self.linear_layer2 = torch.nn.Linear(self.hidden, self.hidden)
        self.linear_layer3 = torch.nn.Linear(self.hidden, self.n_neurons)

    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.activation(x)
        x = self.linear_layer2(x)
        x = self.activation(x)
        x = self.linear_layer3(x)
        return x

#----------------------------------------------------------------------------------------------------------------------

#------------------------------------------- FORWARD AND BACKWARD STEPS -----------------------------------------------


#FORWARD STEP

def forward_step(xi, omega, p_xi, p_omega, t_index, model):
    with torch.no_grad():
        input = torch.cat([xi[t_index], torch.tensor([input_fun(torch.tensor(t_index * dt))])], dim=0 ).unsqueeze(0)
        p_xi[t_index] = model(input)


    # Compute activations
    a = compute_activations(xi, omega, theta, t_index)

    # Forward step (Euler discretization) for the state and p_omega (which depends on p_xi)
    for i in range(n_neurons):
        xi[t_index+1][i] = xi[t_index][i] + dt * (alpha[i]*(-xi[t_index][i] + activation_fun(a[i])))
        for j in range(n_neurons):
            omega[t_index + 1][i][j] = omega[t_index][i][j] + dt * (-p_omega[t_index][i][j]/(phi*m_omega[i][j]))
            p_omega[t_index + 1][i][j] = p_omega[t_index][i][j] + dt * (-phi * (m_xi[i] * (-xi[t_index][i] + activation_fun(a[i])) * activation_fun_prime(a[i]) * theta[i][j] * xi[t_index][j] ) - p_xi[t_index][i] * alpha[i] * activation_fun_prime(a[i]) * theta[i][j] * xi[t_index][j])


#BACKWARD LEARNING

def backward_learning(xi_0, p_xi_0, omega_0, p_omega_0, t_index, net, window):
    xi_train = torch.zeros(window, n_neurons)
    p_xi_train = torch.zeros(window, n_neurons)
    omega_train = torch.zeros(window, n_neurons, n_neurons)
    p_omega_train = torch.zeros(window, n_neurons, n_neurons)
    signal_train = torch.zeros(window)

    # Initialization
    p_xi_train[-1] = p_xi_0
    p_omega_train[-1] = p_omega_0
    xi_train[-1] = xi_0
    omega_train[-1] = omega_0

    # create dataset of optimal solutions going backward
    for window_index in range(window-1):
        for i in range(n_neurons):
            a = compute_activations(xi_train, omega_train, theta, window-window_index-1)
            # print(xi_train[window-window_index-1][i])
            # print(activation_fun(a[i]))
            xi_train[window-window_index-2][i] = xi_train[window-window_index-1][i] - dt * (alpha[i]*(-xi_train[window-window_index-1][i] + activation_fun(a[i])))
            sum1 = 0
            sum2 = 0
            for k in range(n_neurons):
                sum1 += m_xi[k]*(-xi_train[window - window_index - 1][k] + activation_fun(a[k])) * activation_fun_prime(a[k]) * theta[k][i] * omega_train[window-window_index-1][k][i]
                sum2 += p_xi_train[window - window_index - 1][k] * alpha[k] * activation_fun_prime(a[k]) * theta[k][i] * omega_train[window-window_index-1][k][i]
            p_xi_train[window - window_index - 2][i] = p_xi_train[window - window_index - 1][i] - dt * (-phi * (potential_prime(xi_train[window - window_index - 1],input_fun(torch.tensor((t_index - window_index) * dt)),i) - m_xi[i]*(-xi_train[window - window_index - 1][i] + activation_fun(a[i])) + sum1 ) + alpha[i] * p_xi_train[window - window_index - 1][i] - sum2)
            for j in range(n_neurons):
                omega_train[window - window_index - 2][i][j] = omega_train[window - window_index - 1][i][j] - dt * (-p_omega_train[window - window_index - 1][i][j] / (phi * m_omega[i][j]))
                p_omega_train[window - window_index - 2][i][j] = p_omega_train[window - window_index - 1][i][j] - dt * (-phi * (m_xi[i] * (-xi_train[window - window_index - 1][i] + activation_fun(a[i])) * activation_fun_prime(a[i]) * theta[i][j] * xi_train[window - window_index - 1][j]) - p_xi_train[window - window_index - 1][i] * alpha[i] * activation_fun_prime(a[i]) * theta[i][j] * xi_train[window - window_index - 1][j])


    for window_index in range(window):
        signal_train[window-window_index-1] = input_fun(torch.tensor((t_index - window_index) * dt))

    batch = torch.cat((xi_train,signal_train.unsqueeze(1)),dim = 1)

    labels = p_xi_train

    # evaluate model predictions and the corresponding loss
    update_model(net,batch,labels)


#----------------------------------------------------------------------------------------------------------------------

#------------------------------------------------ UPDATE MODEL PARAMETERS ---------------------------------------------

def update_model(net, batch, labels):
    outputs = net(batch)

    loss = criterion(outputs, labels)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()


#----------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------- MAIN PROGRAM -----------------------------------------------------


if __name__ == "__main__":

    # xf = torch.zeros(n)
    xi_f = 0.1*torch.ones(n, n_neurons)
    # xi_f[0]=torch.tensor([0.5,1])
    xi_f[0][0] = torch.tensor([0.5])
    xi_f[0][-1] = torch.tensor([1])

    p_xi_f = 0.1 * torch.rand(n, n_neurons)
    omega_f = 0.1 * torch.rand(n, n_neurons, n_neurons)
    # omega_f[0] = torch.tensor(np.array([0.25330344, 0.63383847, 0.52898445, 0.85110745]).reshape(2,2))

    p_omega_f = 1. * torch.rand(n, n_neurons, n_neurons)

    weights_norm_array = torch.zeros(len(t_array))


    model = NeuralModel()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    for t in range(len(t_array)-1):
        for epoch in range(epochs):
            x_0 = torch.normal(xi_f[t], std=0.5)
            w_0 = torch.normal(omega_f[t], std=0.5)
            backward_learning(x_0, torch.zeros(n_neurons), w_0, torch.zeros(n_neurons), t+window_size, model, window_size)

        # Average Weights' L2 norm for debugging purposes
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        for par in model.parameters():
            if par.requires_grad:
                weights_norm_array[t] += torch.sqrt(torch.sum(par ** 2))
        weights_norm_array[t] /= total_params

        forward_step(xi_f, omega_f, p_xi_f, p_omega_f, t, model)

        # Reset if x or p is out of the bounding box
        if torch.norm(xi_f[t+1])>threshold or torch.norm(omega_f[t+1])>threshold or torch.norm(p_xi_f[t])>threshold or torch.norm(p_omega_f[t])>threshold:
            xi_f[t+1] = 0.1 * torch.rand(n_neurons)
            p_xi_f[t] = 0.1 * torch.rand(n_neurons)
            omega_f[t + 1] = 0.1 * torch.rand(n_neurons, n_neurons)
            p_omega_f[t + 1] = 0.1 * torch.rand(n_neurons, n_neurons)
            print("Reset!")
        if t % 1000 == 0:
            print(f"Iteration {t}/{n}")

    for par in model.parameters():
        if par.requires_grad:
            weights_norm_array[-1] += torch.sqrt(torch.sum(par ** 2))
    weights_norm_array[-1] /= total_params


#----------------------------------------------------------------------------------------------------------------------

#------------------------------------------------ PLOTS ---------------------------------------------------------------

    # xi_f = torch.reshape(xi_f, (n_neurons, n))
    # pi_xi_f = torch.reshape(p_xi_f, (n_neurons, n))

    plt.figure(0)

    for i in range(n_neurons):
        if i == 0:
            plt.plot(t_array, xi_f[:, i], label=r'$\xi_0$', color='green')
            plt.plot(t_array, p_xi_f[:, i], label=r'$p_{\xi_0}$',color='red')
        elif i == 1:
            plt.plot(t_array, xi_f[:,i], color = "lime", label=r'$\xi_i$')
            plt.plot(t_array, p_xi_f[:,i], color = "orange", label=r'$p_{\xi_i}$')
        else:
            plt.plot(t_array, xi_f[:, i], color="lime")
            plt.plot(t_array, p_xi_f[:, i], color="orange")
    plt.plot(t_array, input_fun(t_array).numpy(), label="Signal", color="cyan")
    plt.plot(t_array, np.zeros(len(t_array)),color="black", linestyle="--")

    plt.ylim((-3,3))
    plt.legend()

    plt.figure(1)
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i==0 and j==0:
                plt.plot(t_array, omega_f[:, i, j], label=r'$\omega_{ij}$', color="green")
                plt.plot(t_array, p_omega_f[:, i, j], label=r'$p_{\omega_{ij}}$', color="orange")
            else:
                plt.plot(t_array, omega_f[:,i,j], color="green")
                plt.plot(t_array, p_omega_f[:, i, j], color="orange")
    plt.plot(t_array, np.zeros(len(t_array)),color="black", linestyle="--")

    plt.ylim((-3,3))
    plt.legend()

    plt.figure(2)
    plt.title("Weights norm forward network")
    plt.plot(t_array, weights_norm_array.detach().numpy(), color="orange")

    plt.show()