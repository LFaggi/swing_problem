import numpy as np
import copy

a_old1 = [np.array([0, 3]), np.array([0,1,2,3])]
a_old2 = [np.array([0, 3]), np.array([0,1,2,3])]
a_old3 = [np.array([0, 3]), np.array([0,1,2,3])]
a_old3 = [np.array([0, 3]), np.array([0,1,2,3])]
a_new = [np.array([0, 4]), np.array([0,1,7,3])]

for i in range(len(a_old1)):
    a_old1[i] = a_new[i]

a_old2 = copy.deepcopy(a_new)
a_old3 = copy.copy(a_new)

a
print(a_old1,a_old2,a_old3)