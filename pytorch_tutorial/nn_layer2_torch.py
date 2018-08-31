import torch

dtype = torch.float
device = torch.device('cuda:0' if torch.cuda.is_available() 'cpu')
print('Device: {}'.format(device))

epoch = 5000
train_n = 100
learning_rate = 1e-8
in_dim, hidden_dim, out_dim = 1000, 500, 10
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


def manual_grad(x, y):
    # forward: hidden layer
    a0 = x.t()
    z1 = w1.mm(a0)
    a1 = z1.clamp(min=0.)           # (hidden_dim, train_n)
    # forward: output layer
    z2 = w2.mm(a1)
    a2 = z2                         # (out_dim, train_n)

    # MSE Loss
    loss = (a2 - y.t()).pow(2).sum().item() / train_n

    # backward: output layer
    a2_grad = 2 * (a2 - y.t())      # (out_dim, train_n)
    z2_grad = a2_grad.clone()
    w2_grad = z2_grad.mm(a1.t())    # (out_dim, hidden_dim)
    # backward: hidden layer
    a1_grad = w2.t().mm(z2_grad)    # (hidden_dim, train_n)
    z1_grad = a1_grad.clone()
    z1_grad[z1 < 0.] = 0
    w1_grad = z1_grad.mm(a0.t())    # (hidden_dim, in_dim)

    return w1_grad, w2_grad, loss


def automatic_grad(x, y):
    # require autograd to track them
    w1.requires_grad_(True)
    w2.requires_grad_(True)
    # clear gradients buffer
    if w1.grad is not None:
        w1.grad.zero_()
    if w2.grad is not None:
        w2.grad.zero_()
    # forward
    # output = w2.mm(w1.mm(x.t()).clamp(min=0.))
    output = w2.mm(ReLU.apply(w1.mm(x.t())))
    # MSE Loss
    loss = (output - y.t()).pow(2).sum()
    # backward
    loss.backward()  # autograd for w1 and w2

    return w1.grad, w2.grad, loss.item()/train_n


# train
for e in range(epoch):
    # w1_grad, w2_grad, loss = manual_grad(x, y)
    w1_grad, w2_grad, loss = automatic_grad(x, y)
    
    # MSE Loss
    print('Epoch: {}, Loss: {}'.format(e, loss))
    
    # update parameters
    with torch.no_grad():
        w1 -= learning_rate * w1_grad
        w2 -= learning_rate * w2_grad