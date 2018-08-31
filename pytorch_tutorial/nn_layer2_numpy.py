import numpy as np


epoch = 5000
train_n = 100
learning_rate = 1e-8
in_dim, hidden_dim, out_dim = 1000, 500, 10
x = np.random.randn(train_n, in_dim)
y = np.random.randn(train_n, out_dim)

w1 = np.random.randn(hidden_dim, in_dim)
w2 = np.random.randn(out_dim, hidden_dim)

for e in range(epoch):
    # Input layer
    a0 = x.T                    # (in_dim, train_n)
    # Hidden layer forward
    z1 = np.dot(w1, a0)         # (hidden_dim, train_n)
    # ReLU activation
    a1 = np.maximum(z1, 0.)     # (hidden_dim, train_n)
    
    # Output layer forward
    z2 = np.dot(w2, a1)         # (out_dim, train_n)
    # Identity activation
    a2 = z2                     # (out_dim, train_n)

    # MSE Loss of train data
    loss = np.sum(np.square(a2 - y.T)) / train_n
    print('Epoch: {}, Loss: {}'.format(e, loss))
    
    # Output layer backward
    # square error's gradient
    a2_grad = 2 * (a2 - y.T)        # (out_dim, train_n)
    z2_grad = a2_grad.copy()        # (out_dim, train_n)
    w2_grad = np.dot(z2_grad, a1.T) # (out_dim, hidden_dim)
    
    # Hidden layer backward
    a1_grad = np.dot(w2.T, z2_grad) # (hidden_dim, train_n)
    z1_grad = a1_grad.copy()        # (hidden_dim, train_n)
    z1_grad[z1 < 0.] = 0
    w1_grad = np.dot(z1_grad, x)    # (hidden_dim, in_dim)

    # update parameters
    w1 -= learning_rate * w1_grad
    w2 -= learning_rate * w2_grad
