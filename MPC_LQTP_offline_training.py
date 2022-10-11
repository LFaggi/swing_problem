import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_bvp
import torch
import torch.nn as nn
import torch.optim as optim

dt = 0.01

T_training = 200
n_training = int(T_training // dt)
epochs = 50
batch_dim = 20

T_test = 200
n_test = int(T_test // dt)



a = 1
b = 1
q = 1
r = 0.2

x_0 = 0.1
p_T = 0


# def signal(t):
#     arr=[]
#     for i in range(len(t)):
#         arr.append(5 * math.sin(2 * np.pi * 0.01 * t[i]))  # define the signal to be tracked
#     return np.array(arr)

def input_fun(t):
    return 5 * np.sin(2 * np.pi * 0.01 * t)

def fun(t, y):
    return np.vstack((a*y[0]-(b**2/r)*y[1], -q*(y[0] - input_fun(t))-a*y[1]))

def bc(ya, yb):
    return np.array([ya[0]-x_0, yb[1]-p_T])


class NeuralModel(torch.nn.Module):
    def __init__(self):
        super(NeuralModel, self).__init__()
        self.hidden = 10
        # self.activation = torch.nn.ReLU()
        self.activation = torch.nn.Tanh()
        self.linear_layer1 = torch.nn.Linear(2, self.hidden)
        self.linear_layer2 = torch.nn.Linear(self.hidden, self.hidden)
        self.linear_layer3 = torch.nn.Linear(self.hidden, self.hidden)
        self.linear_layer4 = torch.nn.Linear(self.hidden, 1)

    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.activation(x)
        x = self.linear_layer2(x)
        x = self.activation(x)
        x = self.linear_layer3(x)
        x = self.activation(x)
        x = self.linear_layer4(x)
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


if __name__ == "__main__":

    ############################################ Training #############################################################

    t_array_train_ficticius = np.linspace(0, T_training * 2, num=n_training * 2, endpoint=True)
    t_array_train = np.linspace(0, T_training, num=n_training, endpoint=True)

    y = np.zeros((2, len(t_array_train_ficticius)))

    sol = solve_bvp(fun, bc, t_array_train_ficticius, y, verbose=2, max_nodes=10000)


    x_train = torch.from_numpy(sol.sol(t_array_train)[0])
    p_train = torch.from_numpy(sol.sol(t_array_train)[1])
    signal_train = torch.from_numpy(input_fun(t_array_train))

    training_data = torch.stack((x_train, signal_train), dim=1).to(torch.float32)
    training_labels = p_train.unsqueeze(dim=1).to(torch.float32)

    dataset = torch.utils.data.TensorDataset(training_data, training_labels)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_dim, shuffle=True)

    model = NeuralModel()
    criterion = nn.MSELoss()
    # criterion = nn.MSELoss(reduction='sum')
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.5f}')
                running_loss = 0.0

    ################################################ Test #############################################################


    t_array_test = np.linspace(T_training, T_training + T_test, num=n_test, endpoint=True)

    xf = 0.1 * torch.ones(n_test)
    xf[0] = x_train[-1]
    pf = torch.zeros(n_test)

    for t_index in range(n_test-1):
        forward_step(xf, pf, t_index, model, offset=n_training)


    plt.figure(0)
    plt.plot(t_array_train, x_train, label="State", color="green")
    plt.plot(t_array_test[:-1], xf[:-1], color="green",linestyle='--')
    plt.plot(t_array_train, p_train, label="Costate", color="red")
    plt.plot(t_array_test[:-1], pf[:-1], color="red",linestyle='--')
    plt.plot(t_array_train, input_fun(t_array_train), color="cyan")
    plt.plot(t_array_test[:-1], input_fun(t_array_test)[:-1], label="Signal", color="cyan")
    plt.ylim((-10, 10))
    plt.axhline(y=0, color='black', linestyle='--')
    plt.axvline(x=t_array_train[-1], color='black', linestyle='--')
    plt.legend()
    plt.savefig("state_costate.pdf", dpi=500)

    plt.show()