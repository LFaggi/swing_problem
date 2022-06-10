import matplotlib.pyplot as plt
import numpy as np

# dot(xi) = alpha( -xi+sigma(w u + b))
# dot(w) = v
# L = diss_function * (0.5 * (xi - u)^2 + 0.5 m_v v^2 + 0.5 m_xi (xi-sigma(w u + b))^2 )

# Hamilton equations
# dot(xi) = alpha( -xi+sigma(w u + b))
# dot(u) =-p_w / (m_v * diss_function)
# dot p_xi = - diss_function * [(xi - u) + m_xi (xi - sigma(w u + b))] + p_xi alpha
# dot p_w = -m_xi (xi - sigma(w u +b)) sigma'(w u + b) u diss_function - p_xi alpha sigma'(w u) u

T = 10
delta_t = 0.01

def input(t):
    return 2  # the solution should be pi/4

class NeuralAgent:
    def __init__(self):
        self.xi = 2
        self.omega = 0. * np.random.rand(1)

        self.p_xi = 0.1
        self.p_omega = 0.1

    @staticmethod
    def activation(x):
        return np.tanh(x)

    @staticmethod
    def activation_prime(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def get_env_parameters(env_parameters):
        dissipation_factor = env_parameters[0]
        alpha = env_parameters[1]
        m_xi = env_parameters[2]
        m_v = env_parameters[3]
        b = env_parameters[4]
        return dissipation_factor, alpha, m_xi, m_v, b

    def update_states_costates(self, env_parameters, input):
        dissipation_factor, alpha, m_xi, m_v, b = self.get_env_parameters(env_parameters)

        xi = self.xi
        omega = self.omega
        p_xi = self.p_xi
        p_omega = self.p_omega

        self.xi = xi + delta_t * alpha * (-xi + self.activation(omega * input + b))
        self.omega = omega - delta_t * p_omega/(m_v * dissipation_factor)

        self.p_xi = p_xi + delta_t *(- dissipation_factor * ((xi - input) + m_xi * (xi - self.activation(omega * input + b))) + p_xi * alpha)
        self.p_omega = p_omega + delta_t * (- dissipation_factor * input * m_xi * (xi - self.activation(omega * input + b)) * self.activation_prime(omega * input + b) - p_xi * alpha * self.activation_prime(omega * input + b) * input)

    def evaluate_current_hamiltonian(self,env_parameters, input):
        dissipation_factor, alpha, m_xi, m_v, b = self.get_env_parameters(env_parameters)

        self.h = dissipation_factor * (0.5 * (self.xi - input)**2 + 0.5 * m_xi * (self.xi - self.activation(self.omega * input + b))**2) - 0.5 * self.p_omega**2 / (m_v * dissipation_factor) + self.p_xi * alpha * (-self.xi + self.activation(self.omega * input + b))

    def evaluate_current_optimal_control(self,env_parameters):
        dissipation_factor, alpha, m_xi, m_v, b = self.get_env_parameters(env_parameters)
        self.control = - self.p_omega / (m_v * dissipation_factor)

class EnviromentalAgent:
    def __init__(self):
        self.env_parameters = [1, 1, 1, 1, 1]

    @staticmethod
    def dissipation_function(t):
        return 1

    @staticmethod
    def m_xi_function(t):
        return 1

    def alpha_function(self, t, agent, b):
        num = self.dissipation_function(t) * input(t) * self.m_xi_function(t) * (agent.xi - agent.activation(agent.omega * input(t) + b)) * agent.activation_prime(agent.omega * input(t) + b)
        den = agent.p_xi * agent.activation_prime(agent.omega * input(t) + b) * input(t)
        if agent.p_omega>=0:
            if den > 0:
                return - num / den + 0.5
            elif den < 0:
                return num / abs(den) - 0.5
        elif agent.p_omega<0:
            if den > 0:
                return - num / den - 0.5
            elif den < 0:
                return num / abs(den) + 0.5

    def b_function(self, t, agent):
        diss = self.dissipation_function(t)
        m_xi = self.m_xi_function(t)
        arg = (diss * (agent.xi - input(t)) + diss * m_xi * agent.xi - agent.p_xi * self.env_parameters[1])/(diss * m_xi)
        if arg>=1:
            arg=0.95
        elif arg <=-1:
            arg = -0.95
        if agent.p_xi >= 0:
            return np.arctanh(arg) - agent.omega * input(t) - 0.5
        elif agent.p_xi < 0:
            return np.arctanh(arg) - agent.omega * input(t) + 0.5


    # def alpha_function(self, t, agent, m_xi):
    #     if agent.p_xi >= 0:
    #         return self.dissipation_function(t) * (agent.xi-input(t) + m_xi * (agent.xi - agent.activation(agent.omega * input(t))))/agent.p_xi - 0.1
    #     elif agent.p_xi<0:
    #         return -self.dissipation_function(t) * (agent.xi-input(t) + m_xi * (agent.xi - agent.activation(agent.omega * input(t))))/abs(agent.p_xi) - 0.1
    #
    # def m_xi_function(self, t, agent):
    #     temp = (agent.xi - agent.activation(agent.omega * input(t)) * agent.activation_prime(agent.omega * input(t)) * input(t) * self.dissipation_function(t))
    #     if agent.p_omega >= 0:
    #         if temp >=0:
    #             return -(agent.p_xi * self.alpha_function(t,agent, self.env_parameters[2]) * input(t) * agent.activation_prime(agent.omega * input(t)))/temp + 0.1
    #         elif temp < 0:
    #             return +(agent.p_xi * self.alpha_function(t,agent, self.env_parameters[2]) * input(t) * agent.activation_prime(agent.omega * input(t)))/abs(temp) - 0.1
    #     if agent.p_omega < 0:
    #         if temp >= 0:
    #             return -(agent.p_xi * self.alpha_function(t, agent, self.env_parameters[2]) * input(
    #                 t) * agent.activation_prime(agent.omega * input(t))) / temp - 0.1
    #         elif temp < 0:
    #             return +(agent.p_xi * self.alpha_function(t, agent, self.env_parameters[2]) * input(
    #                 t) * agent.activation_prime(agent.omega * input(t))) / abs(temp) + 0.1

    @staticmethod
    def m_v_function(t):
        return 1

    def update_env_parameters(self,t, agent):
        # In the form dissipation_factor, alpha, m_xi, m_v
        diss = self.dissipation_function(t)
        m_xi = self.m_xi_function(t)
        b =  self.b_function(t, agent)
        alpha = self.alpha_function(t, agent,b)

        m_v = self.m_v_function(t)

        self.env_parameters = [diss, alpha, m_xi, m_v, b]

        # self.env_parameters = [self.dissipation_function(t), self.alpha_function(t, agent),  self.m_xi_function(t, agent),  self.m_v_function(t)]

if __name__ == "__main__":

    t_array = np.arange(0, T, delta_t)
    agent = NeuralAgent()
    envs = EnviromentalAgent()

    xi_list = []
    omega_list = []
    p_xi_list = []
    p_omega_list = []
    control_list = []
    h_list = []

    # alpha_list = []
    # m_xi_list = []
    # m_omega_list = []
    # diss_factor_list = []

    signal_list = []

    list_of_lists = [xi_list, omega_list, p_xi_list, p_omega_list, control_list, h_list, signal_list]
    # list_hyperparameters = []

    def update_list_of_lists(x):
        for i in range(7):
            list_of_lists[i].append(x[i])

    envs.update_env_parameters(0, agent)
    agent.evaluate_current_optimal_control(envs.env_parameters)
    agent.evaluate_current_hamiltonian(envs.env_parameters, input(0))
    update_list_of_lists([agent.xi, agent.omega, agent.p_xi, agent.p_omega, agent.control, agent.h, input(0)])

    for i in range(1,len(t_array)):
        t = t_array[i]
        envs.update_env_parameters(t, agent)
        agent.update_states_costates(envs.env_parameters, input(t))
        agent.evaluate_current_optimal_control(envs.env_parameters)
        agent.evaluate_current_hamiltonian(envs.env_parameters, input(t))

        print(agent.xi, agent.omega, agent.p_xi, agent.p_omega, agent.control, agent.h, input(t))
        update_list_of_lists([agent.xi, agent.omega, agent.p_xi, agent.p_omega, agent.control, agent.h, input(t)])

plt.figure(0)
plt.plot(t_array,list_of_lists[0], label=r'$\xi$', color="blue")
plt.plot(t_array,list_of_lists[1], label=r'$\omega$', color="red")
plt.plot(t_array,list_of_lists[2], label=r'$p_\xi$', color="blue", linestyle="-.")
plt.plot(t_array,list_of_lists[3], label=r'$p_\omega$', color="red", linestyle="-.")

plt.plot(t_array,list_of_lists[4], label=r'$Control$', color="green")

plt.plot(t_array,list_of_lists[6], label=r'$Input$', color="cyan")

plt.ylim(-5,5)
plt.legend()


plt.figure(1)
plt.title("Hamiltonian")
plt.plot(t_array,list_of_lists[5], label=r'$Hamiltonian$', color="orange")
plt.ylim(-5,5)

plt.show()