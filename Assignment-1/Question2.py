import torch as t
import numpy as np
from sklearn.linear_model import LinearRegression
from torch.autograd import Variable as V

# Extract the dataset
data = np.genfromtxt('qn2_data.csv', delimiter=',')

print(data)

x_data = V(t.from_numpy(data[:,:-1]).type(t.FloatTensor))
y_data = V(t.from_numpy(data[:,-1]).type(t.FloatTensor))
n = data.shape[0]

# loss and optimizer
loss = t.nn.MSELoss()
linear_regress = t.nn.Linear(x_data.size(1), 1)
optimizer = t.optim.SGD(linear_regress.parameters(), lr=1e-03)

for i in range(0, 10):
    cur_x_data, cur_y_data = x_data[i % n], y_data[i % n]  # Stochastic gradient descent, one example per iteration
    cur_loss = loss(linear_regress(cur_x_data), cur_y_data)
    optimizer.zero_grad()
    cur_loss.backward()
    optimizer.step()
    print(cur_loss)

# Actual value from Least Squares regression
x = np.matrix(data[:,:-1])
y = np.matrix(data[:,-1]).T
weights = np.linalg.inv(x.T * x) * x.T * y
print(weights)
