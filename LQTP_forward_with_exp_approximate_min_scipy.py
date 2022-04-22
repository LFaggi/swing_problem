import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

T = 100
lambda_decay = 1.2

a = -1
b = 1
q = 1
r = 0.1

x_0 = 0
p_0 = 0

class LQTP_problem:
    def __init__(self,a,b,q,r,x_0,p_0,T,lambda_decay):
        self.T = T
        self.lambda_decay = lambda_decay

        self.a = a
        self.b = b
        self.q = q
        self.r = r

        self.x_0 = x_0
        self.p_0 = p_0
        self.w_0 = 0.1

        self.y = np.zeros(2)

    def signal(self,t):
        return np.array(math.sin(t)) # define the signal to be tracked

    def sigma(self,w):
        # return w
        return math.tanh(w) # define your own function

    # def hamiltonian(self,w,x,p,t):
    #     h = 0.5 * self.q * math.exp(-self.lambda_decay*(self.T-t)) * (x-self.signal(t))**2 + 0.5 * self.r * w**2 + p * (self.a * x + self.b * self.sigma(w))
    #     print(h)
    #     return h

    def eval_w_min(self,x,p,t):
        ham = lambda w:  0.5 * self.q * math.exp(-self.lambda_decay*(self.T-t)) * (x-self.signal(t))**2 + 0.5 * self.r * w**2 + p * (self.a * x + self.b * self.sigma(w))
        w_min = float(minimize(ham, self.w_0).x)
        self.w_0 = w_min
        print(w_min)
        return w_min

    def fun(self,t, y):
        return [self.a*y[0]+ self.b * self.sigma(self.eval_w_min(y[0],y[1],t)), -self.q*math.exp(-self.lambda_decay*(self.T-t))*(y[0] - self.signal(t))-self.a*y[1]]

    def solve_problem(self):
        return solve_ivp(self.fun, [0,self.T], [self.x_0,self.p_0], dense_output=True, rtol=10**-10, method="DOP853")

problem = LQTP_problem(a,b,q,r,x_0,p_0,T,lambda_decay)

sol = problem.solve_problem()

n_samples = 1000
t_plot = np.linspace(0, T, n_samples)
signal_plot=[]
for i in range(0,n_samples):
    signal_plot.append(problem.signal(t_plot[i]))


plt.plot(t_plot, sol.sol(t_plot)[0], label='State',color="blue")
plt.plot(t_plot, sol.sol(t_plot)[1], label='Costate',color="red")
plt.plot(t_plot, signal_plot, label='Signal',color="green")
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("t")
plt.xlim(0,T)
plt.ylim(-1.1,1.1)
plt.legend()

plt.show()