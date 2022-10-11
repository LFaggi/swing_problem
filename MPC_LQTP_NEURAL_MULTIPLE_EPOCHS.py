import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import argparse
import wandb

parser = argparse.ArgumentParser(description='MPC neural approach for the LQTP')

parser.add_argument('--T', type=float, default=300.)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--window_dataset', type=float, default=0.5)
parser.add_argument('--window_mpc', type=float, default=1)
parser.add_argument('--n_batches', type=int, default=10)
parser.add_argument('--dim_batches', type=int, default=10)
parser.add_argument('--a', type=float, default=1.)
parser.add_argument('--b', type=float, default=1.)
parser.add_argument('--r', type=float, default=0.1)
parser.add_argument('--q', type=float, default=1.)
parser.add_argument('--threshold', type=float, default=0.05)

args = parser.parse_args()



T = args.T   # in seconds
dt = args.dt  # in seconds

window_dataset = args.window_dataset          # in seconds
window_mpc = args.window_mpc                  # in seconds

n_batches = args.n_batches
dim_batches = args.dim_batches

a = args.a
b = args.b
r = args.r
q = args.q

threshold = args.threshold


#----------------------------------------------- DEFINITION OF THE INPUT SIGNAL ---------------------------------------


def input_fun(t):
    return 5 * np.sin(2 * np.pi * 0.01 * t)

#----------------------------------------------------------------------------------------------------------------------

#--------------------------------------- NEURAL NETWORK FOR PREDICTIONG THE COSTATE -----------------------------------


class NeuralModel(torch.nn.Module):
    def __init__(self):
        super(NeuralModel, self).__init__()
        self.hidden = 10
        # self.activation = torch.nn.ReLU()
        self.activation = torch.nn.Tanh()
        self.linear_layer1 = torch.nn.Linear(2, self.hidden)
        self.linear_layer2 = torch.nn.Linear(self.hidden, self.hidden)
        self.linear_layer3 = torch.nn.Linear(self.hidden, 1)

    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.activation(x)
        x = self.linear_layer2(x)
        x = self.activation(x)
        x = self.linear_layer3(x)
        return x


#----------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------- FORWARD STEP -----------------------------------------------------

def forward_step(x, p, t_ind, neural_model, offset=0):
    with torch.no_grad():
        state = x[t_ind].unsqueeze(0)
        sig = torch.tensor(input_fun((t_ind + offset) * dt)).to(torch.float32).unsqueeze(0)
        input_batch = torch.cat((state, sig)).unsqueeze(0)

        p[t_ind] = neural_model(input_batch)   # costate prediction

    # Forward step (Euler discretization) for the state
    x[t_ind+1] = x[t_ind] + dt * (a * x[t_ind]-(b**2/r) * p[t_ind])


#----------------------------------------------------------------------------------------------------------------------

#-------------------------------------------- MPC FUNCTION (FOR SUPERVISION) ------------------------------------------


def fun(t, y):
    return np.vstack((a*y[0]-(b**2/r)*y[1], -q*(y[0] - input_fun(t))-a*y[1]))


def MPC(x0, pT, window, t):
    def bc(ya, yb):
        return np.array([ya[0] - x0, yb[1] - pT])

    temporal_array = np.linspace(t, t + window, num=int(window//dt), endpoint=True)
    y = np.zeros((2, temporal_array.size))
    sol = solve_bvp(fun, bc, temporal_array, y, max_nodes=10000, verbose=0)

    if sol.status != 0:
        print("Some problems with MPC occurred", sol.status)

    xf_next = sol.sol(temporal_array)[0][1]
    pf_next = sol.sol(temporal_array)[1][1]
    p0 = sol.sol(temporal_array)[1][0]
    sig_next = input_fun(temporal_array[1])


    return xf_next, pf_next, p0, sig_next


def generate_dataset(x0, pT, t_ind, w_dataset, w_mpc):
    n_point = int(w_dataset//dt)
    x_dataset = np.zeros(n_point)
    p_dataset = np.zeros(n_point)
    signal_dataset = np.zeros(n_point)
    x_dataset[0] = x0
    signal_dataset[0] = torch.tensor(input_fun(t_ind*dt))

    # create the dataset through MPC
    for i in range(n_point - 1):
        if i == 0:
            x_dataset[i + 1], p_dataset[i + 1], p_dataset[0], signal_dataset[i + 1] = MPC(x_dataset[i], pT, w_mpc, (t_ind + i) * dt)
        else:
            x_dataset[i+1], p_dataset[i+1], _, signal_dataset[i+1] = MPC(x_dataset[i], pT, w_mpc, (t_ind + i) * dt)

    x_train = torch.from_numpy(x_dataset[:int(3 * n_point//4)])
    p_train = torch.from_numpy(p_dataset[:int(3 * n_point//4)])
    signal_train = torch.from_numpy(signal_dataset[:(int(3*n_point//4))])

    x_test = torch.from_numpy(x_dataset[int(3 * n_point // 4):])
    p_test = torch.from_numpy(p_dataset[int(3 * n_point // 4):])
    signal_test = torch.from_numpy(signal_dataset[int(3 * n_point // 4):])

    # training data
    batch_train = torch.stack((x_train, signal_train), dim=1).to(torch.float32)
    labels_train = p_train .unsqueeze(dim=1).to(torch.float32)

    # test data
    batch_test = torch.stack((x_test, signal_test),
                              dim=1).to(torch.float32)
    labels_test = p_test.unsqueeze(dim=1).to(torch.float32)

    return (batch_train, labels_train), (batch_test, labels_test)

#----------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------ UPDATE MODEL PARAMETERS ---------------------------------------


def update_model(net, data):
    batch = data[0]
    labels = data[1]
    outputs = net(batch)

    loss = criterion(outputs, labels)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

#----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------- MAIN PROGRAM ---------------------------------------------------


if __name__ == "__main__":

    # WandB initialization
    wandb.init(project="MPC_LQTP_NEURAL", entity="lfaggi")
    wandb.config = {
        "T": T,
        "dt": dt,
        "window_mpc": window_mpc,
        "window_training": window_dataset,
        "a": a,
        "b": b,
        "r": r,
        "q": q
    }



    n = int(T//dt)
    t_array = np.linspace(0, T, n)

    xf = 0.1 * torch.ones(n)
    pf = torch.zeros(n)

    model = NeuralModel()
    criterion = nn.MSELoss(reduction='sum')
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    weights_norm_array = torch.zeros(len(t_array)-1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Optional
    wandb.watch(model)

    for t_index in range(int(3*n//4)-1):
        # Average Weights' L2 norm
        temp=0
        for par in model.parameters():
            if par.requires_grad:
                temp += torch.sum(par ** 2)
            temp = torch.sqrt(temp)
            weights_norm_array[t_index] = temp / total_params

        train = True
        counter = 0

        dataset_train, dataset_test = generate_dataset(xf[t_index], 0, t_index, window_dataset, window_mpc)
        while train:
            displacement = 0
            state_test = dataset_train[0][-1,0] * torch.ones(dataset_test[0].shape[0])
            costate_test = dataset_train[1][-1] * torch.ones(dataset_test[1].shape[0])
            for k in range(dataset_test[0].shape[0]-1):
                forward_step(state_test, costate_test, k, model, offset=(t_index + dataset_train[0].shape[0]))        # TODO  + t_index + # dati training
                displacement += (state_test[k+1]-dataset_test[0][k+1,0])**2 + (costate_test[k] - dataset_test[1][k])**2
            displacement /= len(state_test)

            if displacement < threshold and counter > 0:
                train = False
            elif counter > 100:
                train = False
            else:
                counter += 1
                print(counter)
                for i in range(n_batches):
                    indices = torch.randperm(dataset_train[0].shape[0] - 1)[:dim_batches]
                    tuple_train = (dataset_train[0][indices], dataset_train[1][indices])
                    update_model(model, tuple_train)

            print("Displacement", float(displacement))

        forward_step(xf, pf, t_index, model)
        wandb.log({"State": xf[t_index], "Costate": pf[t_index], "Signal": input_fun(t_index * dt),
                   "Weights norm": weights_norm_array[t_index]}, step=t_index)
        print(t_index, " out of ", n)

    for t_index in range(int(3*n//4) - 1, n-1):

        temp=0
        for par in model.parameters():
            if par.requires_grad:
                temp += torch.sum(par ** 2)
            temp = torch.sqrt(temp)
            weights_norm_array[t_index] = temp / total_params

        forward_step(xf, pf, t_index, model)

        wandb.log({"State": xf[t_index], "Costate": pf[t_index], "Signal": input_fun(t_index * dt),
                   "Weights norm": weights_norm_array[t_index]}, step=t_index)
        print(t_index, " out of ", n)

    #------------------------------------------------------------------------------------------------------------------

    #------------------------------------------------------ PLOTS -----------------------------------------------------


    plt.figure(0)
    plt.plot(t_array[:-1],xf[:-1], label="State",color="green")
    plt.plot(t_array[:-1], pf[:-1], label="Costate", color="red")
    plt.plot(t_array[:-1],input_fun(t_array)[:-1], label="Signal", color="cyan")
    plt.ylim((-10,10))
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend()
    plt.savefig("state_costate.pdf", dpi=500)

    plt.figure(1)
    plt.plot(t_array[:-1], weights_norm_array.detach().numpy(), label="Weights norm", color="orange")
    plt.legend()
    plt.savefig("weights_norm.pdf", dpi=500)

    plt.show()