import numpy as np

T = 10
delta_t = 0.01

class NeuralAgent:
    def __init__(self):
        self.xi = 0
        self.omega = 0.1 * np.random.rand(1)

        self.p_xi = 0
        self.p_omega = 0


    def activation(self,x):
        return np.tanh(x)

    def update(self,env_parameters, input):
        dissipation_factor = env_parameters[0]
        alpha = env_parameters[1]
        m_xi = env_parameters[2]
        m_v = env_parameters[3]

        xi = self.xi
        omega = self.omega
        p_xi = self.p_xi
        p_omega = self.p_omega

        self.xi = xi + delta_t * alpha * (-xi + self.activation(omega * input))
        self.omega = omega - delta_t * p_omega/(m_v * dissipation_factor)

