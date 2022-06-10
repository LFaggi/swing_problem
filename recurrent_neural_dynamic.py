import numpy as np
import math
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--T", type=float, default=100., help="Time Horizon")
parser.add_argument("--delta_t", type=float, default=0.01, help="Integration step")

parser.add_argument("--n_phi", type=int, default=10)
parser.add_argument("--n_omega", type=int, default=10)
parser.add_argument("--n_control", type=int, default=10)
parser.add_argument("--n_psi", type=int, default=10)
parser.add_argument("--n_hidden", type=int, default=100)

parser.add_argument("--phi_0", type=float, default=0.01, help="Initial angle")
parser.add_argument("--omega_0", type=float, default=0.01, help="Initial velocity")
parser.add_argument("--m", type=float, default=1.)
parser.add_argument("--l", type=float, default=1.)
parser.add_argument("--diss", type=float, default=0.1)

args = parser.parse_args()

number_of_neurons = [args.n_phi, args.n_omega, args.n_control, args.psi, args.n_hidden]

total_n_neurons = 0
for x in number_of_neurons:
    total_n_neurons += x

import matplotlib

if args.on_server == "yes":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

T = args.T
delta_t = args.delta_t

m = args.m
l = args.l
diss = args.diss
g = 9.81


class NeuralAgent:
    def __init__(self):
        w = 0.1 * np.random.rand(total_n_neurons, total_n_neurons)
        self.weights = (w+w.T)/2

        self.neurons = np.zeros(total_n_neurons)
        self.co_neurons = np.zeros(total_n_neurons)
        self.co_weights = np.zeros(total_n_neurons, total_n_neurons)

    def aggregate(self,type):
        output = 0
        if type == "phi":
            for i in range(0,number_of_neurons[0]-1):
                output += self.neurons[i]
            output /= number_of_neurons[0]
        if type == "omega":
            for i in range(number_of_neurons[0],number_of_neurons[1]-1):
                output += self.neurons[i]
            output /= number_of_neurons[1]
        if type == "control":
            for i in range(number_of_neurons[1],number_of_neurons[2]-1):
                output += self.neurons[i]
            output /= number_of_neurons[2]
        if type == "psi":
            for i in range(number_of_neurons[2],number_of_neurons[3]-1):
                output += self.neurons[i]
            output /= number_of_neurons[3]

        return output

    def update(self):
        return 0

class PhysicalModel:
    def __init__(self, args):
        self.phi = args.phi_0
        self.omega = args.omega_0

    def read_system(self):
        return self.phi,self.omega

    def update(self, control):
        phi = self.phi
        omega = self.omega
        self.phi += omega * delta_t
        self.omega += delta_t * (-g * np.sin(phi) - diss * omega + control / m)

class EnviromentalAgent:
    def __init__(self, args):
        self.alpha = np.zeros(total_n_neurons)


        return

    def update_enviromental_parameters(self):
        return

    def read_enviromental_parameters(self):
        return