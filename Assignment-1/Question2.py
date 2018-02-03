import torch as t
import numpy as np
from torch.autograd import Variable as V
from matplotlib import pyplot as plt

# Extract the dataset
data = np.genfromtxt('qn2_data.csv', delimiter=',')
N_ITER = 25000

x_data = V(t.from_numpy(data[:,:-1]).type(t.FloatTensor))
y_data = V(t.from_numpy(data[:,-1]).type(t.FloatTensor))
n = data.shape[0]

print(data)

# loss and optimizer
loss = t.nn.MSELoss()
linear_regress = t.nn.Linear(x_data.size(1), 1)
optimizer = t.optim.SGD(linear_regress.parameters(), lr=1e-03)
loss_tracker = []

for i in range(0, N_ITER):
    cur_x_data, cur_y_data = x_data[i % n], y_data[i % n]  # Stochastic gradient descent, one example per iteration
    cur_loss = loss(linear_regress(cur_x_data), cur_y_data)
    loss_tracker.append([i, cur_loss.data])
    optimizer.zero_grad()
    cur_loss.backward()
    optimizer.step()

loss_tracker = np.array(loss_tracker)
plt.plot(loss_tracker[:,0], loss_tracker[:,1], 'k.', alpha=0.5)
plt.show()

for p in linear_regress.parameters():
    print(p)

# Actual value from Least Squares regression without bias
weights1 = t.inverse(x_data.t().mm(x_data)).mm(x_data.t()).mm(y_data.unsqueeze(-1))
print(weights1)

# Actual value from Least Squares regression with bias
x_data = t.cat([t.ones_like(y_data).unsqueeze(-1), x_data], dim=1)
weights2 = t.inverse(x_data.t().mm(x_data)).mm(x_data.t()).mm(y_data.unsqueeze(-1))
print(weights2)

# Predicted values from SGD linear regression, Least squares regression without bias and with bias
test_set = V(t.Tensor([[6, 4], [10, 5], [14, 8]]))
print("Predicted from Test Set (SGD): \n{0}".format(linear_regress(test_set)))
print("Predicted from Test Set (LSR w/o bias): \n{0}".format(test_set.mm(weights1)))
print("Predicted from Test Set (LSR w/ bias): \n{0}".format(test_set.mm(weights2[1:]) + weights2[0]))
