import torch

dtype = torch.float
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

epoch = 5000
train_n = 100
learning_rate = 1e-8
in_dim, hidden_dim, out_dim = 1000, 500, 10

# Setting requires_grad=False(default) indicates that we do not need to 
# compute gradients with respect to these Tensors during the backward pass.
x = torch.randn(train_n, in_dim, dtype=dtype, device=device)
y = torch.randn(train_n, out_dim, dtype=dtype, device=device)

# init parameters
w1 = torch.randn(hidden_dim, in_dim, dtype=dtype, device=device)
w2 = torch.randn(out_dim, hidden_dim, dtype=dtype, device=device)


class ReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, in_data):
        ctx.save_for_backward(in_data)
        return in_data.clamp(min=0.)

    @staticmethod
    def backward(ctx, out_grad):
        in_data, = ctx.saved_tensors
        in_grad = out_grad.clone()
        in_grad[in_data < 0.] = 0.
        return in_grad


def train_with_manual_grad(x, y, w1, w2, learning_rate):
    # Forward
    # keep the intermediate values since we'll reference them in the backward
    # propagation.
    # forward: hidden layer
    a0 = x.t()
    z1 = w1.mm(a0)
    a1 = z1.clamp(min=0.)           # (hidden_dim, train_n)
    # forward: output layer
    z2 = w2.mm(a1)
    a2 = z2                         # (out_dim, train_n)

    # MSE Loss
    loss = (a2 - y.t()).pow(2).sum().item() / train_n

    # Backward
    # backward: output layer
    a2_grad = 2 * (a2 - y.t())      # (out_dim, train_n)
    z2_grad = a2_grad.clone()
    w2_grad = z2_grad.mm(a1.t())    # (out_dim, hidden_dim)
    # backward: hidden layer
    a1_grad = w2.t().mm(z2_grad)    # (hidden_dim, train_n)
    z1_grad = a1_grad.clone()
    z1_grad[z1 < 0.] = 0
    w1_grad = z1_grad.mm(a0.t())    # (hidden_dim, in_dim)

    # Update parameters
    w1 -= lr * w1_grad
    w2 -= lr * w2_grad

    return loss


def train_with_automatic_grad(x, y, w1, w2, lr):
    # require to compute gradients with respect to these Tensors during the 
    # backward pass.
    w1.requires_grad_(True)
    w2.requires_grad_(True)
    # clear gradients buffer
    if w1.grad is not None:
        w1.grad.zero_()
    if w2.grad is not None:
        w2.grad.zero_()
    # Forward
    # we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    # output = w2.mm(w1.mm(x.t()).clamp(min=0.))
    output = w2.mm(ReLU.apply(w1.mm(x.t())))
    # MSE Loss
    loss = (output - y.t()).pow(2).sum()
    # Backward
    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Update parameters
    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # 
    # w1.data -= learning_rate * w1.grad
    # w2.data -= learning_rate * w2.grad
    # 
    # You can also use torch.optim.SGD to achieve this.

    return loss.item() / train_n


# train
for e in range(epoch):
    loss = train_with_automatic_grad(x, y, w1, w2, learning_rate)
    # loss = train_with_manual_grad(x, y, w1, w2, learning_rate)    
    # MSE Loss
    print('Epoch: {}, Loss: {}'.format(e, loss))
    