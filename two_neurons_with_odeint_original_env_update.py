import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy.integrate
np.set_printoptions(precision=10)

class NeuralAgent:
    def __init__(self):

        self.n_neurons = 2
        self.xi = np.array([0.5,1.])
        self.omega = np.array([[0.5507979025745755, 0.7081478226181048], [0.2909047389129443, 0.510827605197663]])

        self.p_xi = np.array([-0.352496578521608, -0.044991618497037325])
        self.p_omega = np.array([[-0.0218882036703026, -0.03619190031163536], [-0.004876276476018469, -0.01111928939200455]])

        self.agent_state = [self.xi, self.omega, self.p_xi, self.p_omega]

        self.xi_list = []
        self.omega_list = []
        self.p_xi_list = []
        self.p_omega_list = []
        self.h_list =[]
        self.p_norm_list = []

        self.history = [self.xi_list, self.omega_list, self.p_xi_list, self.p_omega_list, self.h_list, self.p_norm_list]

        self.threshold2 = 0.

        self.h = 0
        self.p_norm = 0

    def update_history(self):
        self.xi_list.append(copy.deepcopy(self.xi))
        self.omega_list.append(copy.deepcopy(self.omega.reshape(self.n_neurons**2)))
        self.p_xi_list.append(copy.deepcopy(self.p_xi))
        self.p_omega_list.append(copy.deepcopy(self.p_omega.reshape(self.n_neurons**2)))
        self.h_list.append(copy.deepcopy(self.h))
        self.p_norm_list.append(copy.deepcopy(self.p_norm))


    def set_input(self,input):
        self.input = input

    @staticmethod
    def activation_fun(x):
        return np.tanh(x)

    @staticmethod
    def activation_fun_prime(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def activation_fun_second(x):
        return -2 * np.tanh(x) * (1/np.cosh(x))**2

    def potential(self,xi,u):
        return 0.5 * (xi[0] - u)**2

    def potential_prime(self,xi,u,i):
        if i == 0:
            return (xi[0] - u)
        elif i == 1:
            return 0

    def read_env_parameters(self,env_parameters):
        self.dissipation_factor = env_parameters[0]
        self.alpha = env_parameters[1]
        self.theta = env_parameters[2]
        self.m_xi = env_parameters[3]
        self.m_v = env_parameters[4]

    def compute_activations(self):
        activations = np.zeros(self.n_neurons)
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                activations[i] += self.theta[i, j] * self.omega[i, j] * self.xi[j]
        return activations

    def right_hand(self,t,y): # FOr scipy odeint
        def input_fun(x):
            return 1. * np.sin(x) + 0.5

        alpha = self.alpha
        phi = self.dissipation_factor
        theta = np.array([self.theta[0,0], self.theta[0,1], self.theta[1,0], self.theta[1,1]])
        mv = self.m_v[0,0] #sono tutti uguali

        a1 = theta[0] * y[2] * y[0] + theta[1] * y[3] * y[1]
        a2 = theta[2] * y[4] * y[0] + theta[3] * y[5] * y[1]

        activations = np.array([a1, a2])

        a = np.array([(alpha[0] * (-y[0] + self.activation_fun(a1)),
                          alpha[1] * (-y[1] + self.activation_fun(a2)),
                          -y[8] / (phi * mv),
                          -y[9] / (phi * mv),
                          -y[10] / (phi * mv),
                          -y[11] / (phi * mv),
                          -phi * (y[0] - input_fun(t)) + alpha[0] * y[6] - alpha[0] * y[6] * self.activation_fun_prime(a1) *
                          theta[0] * y[2] - alpha[1] * y[7] * self.activation_fun_prime(a2) * theta[2] * y[4],
                          alpha[1] * y[7] - alpha[0] * y[6] * self.activation_fun_prime(a1) * theta[1] * y[3] - alpha[1] * y[
                              7] * self.activation_fun_prime(a2) * theta[3] * y[5],
                          - alpha[0] * y[6] * self.activation_fun_prime(a1) * theta[0] * y[0],
                          - alpha[0] * y[6] * self.activation_fun_prime(a1) * theta[1] * y[1],
                          - alpha[1] * y[7] * self.activation_fun_prime(a2) * theta[2] * y[0],
                          - alpha[1] * y[7] * self.activation_fun_prime(a2) * theta[3] * y[1])])
        return np.squeeze(a)

    def update_states_costates(self,t,dt):

        # y = [xi1, xi2, w11, w12, w21, w22, pxi1, pxi2, pw11, pw12, pw21, pw22]
        #       0    1    2    3    4    5     6     7    8      9    10    11

        t_eval = np.linspace(t,t+dt,num=1000,endpoint=True)

        y0 = np.array([self.xi[0], self.xi[1], self.omega[0,0], self.omega[0,1], self.omega[1,0], self.omega[1,1], self.p_xi[0], self.p_xi[1], self.p_omega[0,0], self.p_omega[0,1], self.p_omega[1,0], self.p_omega[1,1]])

        sol = scipy.integrate.solve_ivp(self.right_hand, (t,t+dt), y0, t_eval = t_eval)
        self.xi[0] = sol.y[0][-1]
        self.xi[1] = sol.y[1][-1]
        self.omega[0, 0] = sol.y[2][-1]
        self.omega[0, 1] = sol.y[3][-1]
        self.omega[1, 0] = sol.y[4][-1]
        self.omega[1, 1] = sol.y[5][-1]
        self.p_xi[0] = sol.y[6][-1]
        self.p_xi[1] = sol.y[7][-1]
        self.p_omega[0, 0] = sol.y[8][-1]
        self.p_omega[0, 1] = sol.y[9][-1]
        self.p_omega[1, 0] = sol.y[10][-1]
        self.p_omega[1, 1] = sol.y[11][-1]

    def evaluate_costate_derivative(self,xi,omega,p_xi,p_omega,activations):
        g_p_xi = np.zeros(self.n_neurons)
        g_p_omega = np.zeros((self.n_neurons, self.n_neurons))

        # shortcut
        alpha = self.alpha
        dissipation_factor = self.dissipation_factor
        theta = self.theta
        m_xi = self.m_xi
        m_v = self.m_v

        temp1 = np.zeros(self.n_neurons)
        temp2 = np.zeros(self.n_neurons)

        for k in range(self.n_neurons):
            for i in range(self.n_neurons):
                temp1[k] += m_xi[i] * (-xi[i] + self.activation_fun(activations[i])) * self.activation_fun_prime(
                    activations[i]) * theta[i, k] * omega[i, k]
                temp2[k] += p_xi[i] * alpha[i] * self.activation_fun_prime(activations[i]) * theta[i, k] * omega[i, k]

        for k in range(self.n_neurons):
            g_p_xi[k] = - dissipation_factor * (self.potential_prime(xi,self.input,k)-m_xi[k]*(-xi[k]+self.activation_fun(activations[k])) + temp1[k]) + p_xi[k] * alpha[k] - temp2[k]

            for n in range(self.n_neurons):
                g_p_omega[k, n] = -dissipation_factor * (m_xi[k] * (-xi[k]+self.activation_fun(activations[k])) * self.activation_fun_prime(activations[k]) * theta[k,n] * xi[n]) - p_xi[k] * alpha[k] * self.activation_fun_prime(activations[k]) * theta[k,n] * xi[n]

        return g_p_xi,g_p_omega

    # def evaluate_current_optimal_control(self,env_parameters):
    #     dissipation_factor, alpha, m_xi, m_v, b = self.get_env_parameters(env_parameters)
    #     self.control = - self.p_omega / (m_v * dissipation_factor)

    def evaluate_current_hamiltonian(self):
        activations = self.compute_activations()
        # shortcut
        alpha = self.alpha
        dissipation_factor = self.dissipation_factor
        theta = self.theta
        m_xi = self.m_xi
        m_v = self.m_v

        temp1 = 0
        temp2 = 0
        temp3 = 0
        for i in range(self.n_neurons):
            temp1 += 0.5 * m_xi[i] * (-self.xi[i]+self.activation_fun(activations[i]))**2
            temp2 += self.p_xi[i] * alpha[i] * (-self.xi[i]+self.activation_fun(activations[i]))
            for j in range(self.n_neurons):
                temp3 += 0.5 * self.p_omega[i,j] / (dissipation_factor * m_v[i,j]+self.threshold2)

        self.h = dissipation_factor * (self.potential(self.xi, self.input) + temp1) + temp2 - temp3

    def evaluate_norm_p(self):
        temp = 0
        for i in range(self.n_neurons):
            temp += self.p_xi[i] * self.p_xi[i]
            for j in range(self.n_neurons):
                temp += self.p_omega[i, j] * self.p_omega[i, j]

        self.p_norm = np.sqrt(temp)



class EnviromentalAgent:
    def __init__(self):
        self.n_neurons = 2

        self.dissipation_factor = 1
        self.alpha = np.ones(self.n_neurons) # the initialization of alpha is 0 when it is not fixed
        self.theta = np.ones((self.n_neurons,self.n_neurons)) # np.random.rand(self.n_neurons, self.n_neurons)

        self.m_v = 0.01 * np.ones((self.n_neurons, self.n_neurons))
        self.m_xi = 0. * np.ones(self.n_neurons)


        self.eta_in = 1e-04     # This is a default value
        self.eta_out = 1e-02    # This is a default value
        self.max_it = 5         # This is a default value
        self.threshold = 0.1    # alpha = 0 in the initialization, the gradient in that direction blow up without this threshold!
        self.epsilon = 0.01     # This is a default value

        self.env_parameters = [self.dissipation_factor, self.alpha, self.theta, self.m_xi, self.m_v]

        self.dissipation_factor_list = []
        self.alpha_list = []
        self.theta_list = []
        self.m_xi_list = []
        self.m_v_list = []
        self.slack_list = []

        self.history = [self.dissipation_factor_list, self.alpha_list, self.theta_list, self.m_xi_list, self.m_v_list, self.slack_list]

    def update_history(self):
        dissipation_factor = self.env_parameters[0]
        alpha = self.env_parameters[1]
        theta = self.env_parameters[2]
        m_xi = self.env_parameters[3]
        m_v = self.env_parameters[4]

        self.dissipation_factor_list.append(copy.deepcopy(dissipation_factor))
        self.alpha_list.append(copy.deepcopy(alpha))
        self.theta_list.append(copy.deepcopy(theta.reshape(self.n_neurons ** 2)))
        self.m_xi_list.append(copy.deepcopy(m_xi))
        self.m_xi_list.append(copy.deepcopy(m_v.reshape(self.n_neurons ** 2)))

    def env_potential(self):
        #when phi and alpha are fixed, the environmental loss includes only the term with theta
        temp = 0
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                temp += (self.theta[i, j]-1)**2
        return 0.5 * temp

    def grad_env_potential(self, direction, idx1=None, idx2=None):
        if direction == "theta":
            return self.theta[idx1, idx2]-1

    def grad_g(self, name_constraint, direction, agent, idxc1 = None, idxc2 = None, idx1 = None, idx2=None):

        agent.read_env_parameters(self.env_parameters)
        activations = agent.compute_activations()

        if name_constraint == "g_p_xi":
            # if direction == "dissipation_factor":
            #     temp1 = np.zeros(self.n_neurons)
            #     for i in range(self.n_neurons):
            #         temp1[idxc1] += self.m_xi[i] * (
            #                     -agent.xi[i] + agent.activation_fun(activations[i])) * agent.activation_fun_prime(
            #             activations[i]) * self.theta[i, idxc1] * agent.omega[i, idxc1]
            #     return -(agent.potential_prime(agent.xi, agent.input, idxc1) - self.m_xi[idxc1] * (-agent.xi[idxc1] + agent.activation_fun(activations[idxc1])) + temp1[idxc1])
            # if direction == "alpha":
            #     temp = 0
            #     if idxc1 == idx1:
            #         temp += agent.p_xi[idxc1]
            #     temp += -agent.p_xi[idx1] * agent.activation_fun_prime(activations[idx1]) * self.theta[idx1,idxc1] * agent.omega[idx1,idxc1]
            #     return temp
            if direction == "theta":
                temp = 0
                if idxc1 == idx1:
                    temp += self.dissipation_factor * self.m_xi[idxc1] * agent.activation_fun_prime(activations[idxc1]) * agent.omega[idxc1,idx2] * agent.xi[idx2]
                temp += -self.dissipation_factor * (self.m_xi[idx1] * (agent.activation_fun_prime(activations[idx1]))**2 * agent.omega[idx1,idx2] * agent.xi[idx2] * self.theta[idx1,idxc1] * agent.omega[idx1,idxc1])
                temp += -self.dissipation_factor * (self.m_xi[idx1] * (-agent.xi[idx1] + agent.activation_fun(activations[idx1])) * agent.activation_fun_second(activations[idx1]) * agent.omega[idx1,idx2] * agent.xi[idx2] * self.theta[idx1,idxc1] * agent.omega[idx1,idxc1])
                if idxc1 == idx2:
                    temp += -self.dissipation_factor * (self.m_xi[idx1] * (-agent.xi[idx1] + agent.activation_fun(activations[idx1])) * agent.activation_fun_prime(activations[idx1]) * agent.omega[idx1,idx2])
                temp += - agent.p_xi[idx1] * self.alpha[idx1] * agent.activation_fun_second(activations[idx1]) * agent.omega[idx1, idx2] * agent.xi[idx2] * self.theta[idx1, idxc1] * agent.omega[idx1, idxc1]
                if idxc1 == idx2:
                    temp += - agent.p_xi[idx1] * self.alpha[idx1] * agent.activation_fun_prime(activations[idx1]) * agent.omega[idx1, idx2]
                return temp

        if name_constraint == "g_p_omega":
            # if direction == "dissipation_factor":
            #     return -(self.m_xi[idxc1] *(-agent.xi[idxc1]+agent.activation_fun(activations[idxc1])) * agent.activation_fun_prime(activations[idxc1]) * self.theta[idxc1,idxc2] * agent.xi[idxc2])
            # if direction == "alpha":
            #     if idxc1 == idx1:
            #         return -agent.p_xi[idxc1] * agent.activation_fun_prime(activations[idxc1]) * self.theta[idxc1,idxc2] * agent.xi[idxc2]
            #     else: return 0
            if direction == "theta":
                temp = 0
                if idxc1 == idx1:
                    temp += -self.dissipation_factor * (self.m_xi[idxc1] * (agent.activation_fun_prime(activations[idxc1]))**2 * agent.omega[idxc1, idx2] * self.theta[idxc1, idxc2] * agent.xi[idx2] * agent.xi[idxc2])
                    temp += -self.dissipation_factor * (self.m_xi[idxc1] * (-agent.xi[idxc1] + agent.activation_fun(activations[idxc1])) * agent.activation_fun_second(activations[idxc1]) * agent.omega[idxc1, idx2] * self.theta[idxc1, idxc2] * agent.xi[idx2] * agent.xi[idxc2])
                    if idxc2 == idx2:
                        temp += -self.dissipation_factor * (self.m_xi[idxc1] * (-agent.xi[idxc1] + agent.activation_fun(activations[idxc1])) * agent.activation_fun_prime(activations[idxc1]) * agent.xi[idxc2])
                    temp += - agent.p_xi[idxc1] * self.alpha[idxc1] * agent.activation_fun_second(activations[idxc1]) * agent.omega[idxc1,idx2] * self.theta[idxc1,idxc2] * agent.xi[idx2] * agent.xi[idxc2]
                    if idxc2 == idx2:
                        temp += - agent.p_xi[idxc1] * self.alpha[idxc1] * agent.activation_fun_prime(activations[idxc1]) * agent.xi[idxc2]
                    return temp
                else: return 0

    def grad_constraint(self, direction, agent, idx1 = None, idx2 = None):
        temp = 0
        # if direction == "dissipation_factor":
        #     for i in range(self.n_neurons):
        #         temp += agent.p_xi[i] * self.grad_g("g_p_xi","dissipation_factor", agent, idxc1=i)
        #     for i in range(self.n_neurons): # non distinguo caso simmetrico e non tanto p_omega è triangolare
        #         for j in range(self.n_neurons):
        #             temp += agent.p_omega[i,j] * self.grad_g("g_p_omega","dissipation_factor", agent, idxc1=i, idxc2=j)
        # if direction == "alpha":
        #     for i in range(self.n_neurons):
        #         temp += agent.p_xi[i] * self.grad_g("g_p_xi", "alpha", agent, idxc1=i, idx1=idx1)
        #     for i in range(self.n_neurons):
        #         for j in range(self.n_neurons):
        #             temp += agent.p_omega[i, j] * self.grad_g("g_p_omega", "alpha", agent, idxc1=i, idxc2=j, idx1=idx1)
        if direction == "theta":
            for i in range(self.n_neurons):
                temp += agent.p_xi[i] * self.grad_g("g_p_xi", "theta", agent, idxc1=i, idx1=idx1, idx2=idx2)
            for i in range(self.n_neurons):
                for j in range(self.n_neurons):
                    temp += agent.p_omega[i, j] * self.grad_g("g_p_omega", "theta", agent, idxc1=i, idxc2=j,
                                                              idx1=idx1, idx2=idx2)
        return temp

    def check_feasible(self, agent, env_parameters):
        agent.read_env_parameters(env_parameters)
        activations = agent.compute_activations()

        g_p_xi, g_p_omega = agent.evaluate_costate_derivative(agent.xi, agent.omega, agent.p_xi, agent.p_omega, activations)
        temp = 0
        for i in range(self.n_neurons):
            temp += agent.p_xi[i] * g_p_xi[i]
            for j in range(self.n_neurons):
                temp += agent.p_omega[i,j] * g_p_omega[i, j]
        return temp

    def gradG2(self,agent):
        temp=0
        #temp += self.grad_constraint("dissipation_factor", agent)**2
        for i in range(self.n_neurons):
            #temp += self.grad_constraint("alpha", agent,idx1=i) ** 2
            for j in range(self.n_neurons):
                temp += self.grad_constraint("theta", agent , idx1=i, idx2=j) ** 2

        return temp

    def gradUscalargradG(self,agent):
        temp=0
        #temp += self.grad_constraint("dissipation_factor", agent) * self.grad_env_potential("dissipation_factor")
        for i in range(self.n_neurons):
            #temp += self.grad_constraint("alpha", agent, idx1=i) * self.grad_env_potential("alpha", idx1=i)
            for j in range(self.n_neurons):
                temp += self.grad_constraint("theta", agent, idx1=i, idx2=j) * self.grad_env_potential("theta", idx1=i, idx2=j)

        return temp

    def update_env_parameters(self, agent):

        epsilon = self.epsilon
        max_it = self.max_it
        it = 0
        eta_in = self.eta_in
        eta_out = self.eta_out

        gamma = 2
        # print(self.env_parameters)
        env_parameters = copy.deepcopy(self.env_parameters)
        env_parameters_new = copy.deepcopy(env_parameters)
        a = self.check_feasible(agent, env_parameters)

        print("a", a)
        if a < -epsilon:  # I am in the feasible region
            env_parameters_new[0] = env_parameters[0] - eta_in * self.grad_env_potential("dissipation_factor")
            for i in range(
                    self.n_neurons):  # TODO stava qui il problema modificare la loss rispetto alpha - risolto con soglia al denominatore
                env_parameters_new[1][i] = env_parameters[1][i] - eta_in * self.grad_env_potential("alpha", idx1=i)
                for j in range(self.n_neurons):
                    env_parameters_new[2][i, j] = env_parameters[2][i, j] - eta_in * self.grad_env_potential("theta",
                                                                                                             idx1=i,
                                                                                                             idx2=j)

            b = self.check_feasible(agent, env_parameters_new)
            print("b", b)
            while True:
                if b < -epsilon:
                    # self.env_parameters = copy.deepcopy(env_parameters_new)
                    for i in range(len(self.env_parameters)):
                        self.env_parameters[i] = env_parameters_new[i]
                    break
                else:
                    norm_factor = self.gradUscalargradG(agent) / self.gradG2(agent)
                    env_parameters_new[0] = env_parameters[0] - eta_in * (
                            self.grad_env_potential("dissipation_factor") - gamma * norm_factor * self.grad_constraint(
                        "dissipation_factor", agent))
                    for i in range(self.n_neurons):
                        env_parameters_new[1][i] = env_parameters[1][i] - eta_in * (
                                self.grad_env_potential("alpha", idx1=i) - gamma * norm_factor * self.grad_constraint(
                            "alpha", agent, idx1=i))
                        for j in range(self.n_neurons):
                            env_parameters_new[2][i, j] = env_parameters[2][i, j] - eta_in * (
                                    self.grad_env_potential("theta", idx1=i,
                                                            idx2=j) - gamma * norm_factor * self.grad_constraint(
                                "alpha",
                                agent,
                                idx1=i,
                                idx2=j))

                c = self.check_feasible(agent, env_parameters_new)
                print("c:", c)
                if c < -epsilon:
                    # self.env_parameters = copy.deepcopy(env_parameters_new)
                    for i in range(len(self.env_parameters)):
                        self.env_parameters[i] = env_parameters_new[i]
                    break
                else:
                    print("Lr is too high along the tangential direction!")
                    eta_in /= 2
                    it += 1
                    if it == max_it:
                        # self.env_parameters = copy.deepcopy(env_parameters_new)
                        for i in range(len(self.env_parameters)):
                            self.env_parameters[i] = env_parameters_new[i]
                        break
        if a > -epsilon:  # I am out the feasibile region!
            while True:
                env_parameters_new[0] = env_parameters[0] - eta_out * self.grad_constraint("dissipation_factor", agent)
                for i in range(self.n_neurons):
                    env_parameters_new[1][i] = env_parameters[1][i] - eta_out * self.grad_constraint("alpha", agent,
                                                                                                     idx1=i)
                    for j in range(self.n_neurons):
                        env_parameters_new[2][i, j] = env_parameters[2][i, j] - eta_out * self.grad_constraint("alpha",
                                                                                                               agent,
                                                                                                               idx1=i,
                                                                                                               idx2=j)
                b = self.check_feasible(agent, env_parameters_new)
                print("b", b)
                if b < -epsilon:
                    # self.env_parameters = copy.deepcopy(env_parameters_new)
                    for i in range(len(self.env_parameters)):
                        self.env_parameters[i] = env_parameters_new[i]
                    break
                else:
                    print("Lr is too low along nabla g to come back in the feasible region!")
                    eta_out *= 5
                    it += 1
                    if it == max_it:
                        # self.env_parameters = copy.deepcopy(env_parameters_new)
                        for i in range(len(self.env_parameters)):
                            self.env_parameters[i] = env_parameters_new[i]
                        break

        # if a > -epsilon:        # I am out the feasibile region!
        #     while True:
        #         # print(self.env_parameters[2])
        #         env_parameters = copy.deepcopy(self.env_parameters)
        #         for i in range(self.n_neurons):
        #             for j in range(self.n_neurons):
        #                 self.env_parameters[2][i, j] = env_parameters[2][i, j] - eta_in * (
        #                     self.grad_env_potential("theta", idx1=i, idx2=j)) - eta_out * self.grad_constraint("theta",
        #                                                                                                        agent,
        #                                                                                                        idx1=i,
        #                                                                                                        idx2=j)
        #         b = self.check_feasible(agent, self.env_parameters)
        #         # print(self.env_parameters[2])
        #         print("b", b)
        #         if b < -epsilon:
        #             break
        #         else:
        #             it += 1
        #             if it % 10 == 0:
        #                 eta_out *= 2
        #                 print("eta_out: ", eta_out)
        #             if it == max_it:
        #                 break


        # print(self.env_parameters)
        self.dissipation_factor = self.env_parameters[0]
        self.alpha = self.env_parameters[1]
        self.theta = self.env_parameters[2]
        self.m_xi = self.env_parameters[3]
        self.m_v = self.env_parameters[4]


if __name__ == "__main__":
    np.random.seed(3)
    T = 20
    delta_t = 0.001

    number_of_neurons = 2

    def input_fun(t):
        return 1. * np.sin(t) + 0.5

    signal_list = []

    t_array = np.arange(0, T, delta_t)

    agent = NeuralAgent()
    env_agent = EnviromentalAgent()
    # env_agent.eta_in = 0.0001
    # env_agent.eta_out = 0.001
    env_agent.eta_in = 0.0001
    env_agent.eta_out = 0.0001
    env_agent.max_it = 200
    env_agent.epsilon = 0.

    env_agent.update_history()

    agent.read_env_parameters(env_agent.env_parameters)
    agent.set_input(input_fun(0))

    # Initialization consistent with input
    # agent.xi[0] = 0
    # agent.xi[1] = 0.2

    agent.evaluate_norm_p()
    agent.evaluate_current_hamiltonian()
    agent.update_history()

    for t in t_array:
        print("t:> ", t ,"---------------------------------------------------------")
        input = input_fun(t)
        signal_list.append(input)
        agent.set_input(input)

        env_agent.update_env_parameters(agent)

        if t != t_array[-1]:
            env_agent.update_history()
        agent.read_env_parameters(env_agent.env_parameters)
        agent.update_states_costates(t,delta_t)
        agent.evaluate_norm_p()
        agent.evaluate_current_hamiltonian()
        if t != t_array[-1]:
            agent.update_history()

    # Plotting
    ThetaTimesOmega = []
    for i in range(len(t_array)):
        ThetaTimesOmega.append(agent.history[1][i]*env_agent.history[2][i])

    a = list(zip(*agent.history[0]))

    plt.figure(0)
    plt.plot(t_array, a[0], label=r'$\xi_0$', color="green")
    plt.plot(t_array, a[1], label=r'$\xi_1$', color="red")

    plt.plot(t_array, agent.history[1], label=r'$\omega$', color="orange")
    # plt.plot(t_array, ThetaTimesOmega, label=r'$\theta \cdot \omega$', color="yellow")
    plt.plot(t_array, agent.history[2], label=r'$p_\xi$', color="blue", linestyle="-.")
    plt.plot(t_array, agent.history[3], label=r'$p_\omega$', color="orange", linestyle="-.")
    plt.plot(t_array, np.zeros(len(t_array)),color="black", linestyle="--")

    plt.plot(t_array,signal_list, label=r'$Input$', color="cyan")
    plt.ylim(-1.1,4.1)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("Soluzione forward")


    plt.figure(1)
    plt.title("Hamiltonian")
    plt.plot(t_array,agent.history[4], label=r'$Hamiltonian$', color="orange")
    plt.ylim(-3, 3)

    plt.figure(2)
    plt.title("Enviromental parameters")
    plt.plot(t_array,env_agent.history[0], label=r'$\Phi$',color="blue")
    plt.plot(t_array,env_agent.history[1], label=r'$\alpha_{i}$',color="red")
    plt.plot(t_array,env_agent.history[2], label=r'$\theta_{ij}$',color="green")

    plt.plot(t_array, np.zeros(len(t_array)),color="black", linestyle="--")

    plt.ylim(-2, 2)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.figure(3)
    plt.title("Costates norm")
    plt.plot(t_array,agent.history[5], label=r'$\| p \|$',color="red")

    plt.show()