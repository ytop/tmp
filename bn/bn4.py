import numpy as np
import mindspore

from mindspore import context
from mindspore.common.tensor import Tensor
import mindspore.ops.operations as P
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.nn import Cell
from mindspore.nn import BatchNorm2d
import torch
import torchvision

class Batchnorm_Net(Cell):
    def __init__(self, c, weight, bias, moving_mean, moving_var_init, use_batch_statistics=None):
        super(Batchnorm_Net, self).__init__()
        self.bn = BatchNorm2d(c, eps=0.00001, momentum=0.1, beta_init=bias, gamma_init=weight,
                              moving_mean_init=moving_mean, moving_var_init=moving_var_init,
                              use_batch_statistics=use_batch_statistics)

    def construct(self, input_data):
        x = self.bn(input_data)
        return x


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        gout = self.grad(self.network)(input_data, sens)
        return gout

class TorchNet(torch.nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.bn = torch.nn.BatchNorm2d(3, affine=False)

    def forward(self, x):
        return self.bn(x)

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)
np.random.seed(1)
x_np = np.random.randn(1, 3, 2, 2).astype(np.float32)

# numpy 
x = torch.from_numpy(x_np)
x.requires_grad=True
mean = x.mean(dim=[0, 2, 3], keepdim=True)
invstd = torch.sqrt(x.var([0, 2, 3], unbiased=False, keepdim=True) + 1e-5)
print("x ",x)
print("mean ",mean)
print("invstd ",invstd)
y = (x - mean) / invstd 
print("y ",y)
y.abs().sum().backward()
print("numpy grad  ", x.grad)
print("  ")
x1_grad = x.grad.clone()
y1 = y.clone()

# torch 
x = torch.from_numpy(x_np)
x.requires_grad=True
torch_net = TorchNet()
torch_out = torch_net(x)
torch_out.abs().sum().backward()
torch_out_grad_np = x.grad.data.clone()
print("torch grad ", torch_out_grad_np)

#  mindspore 
x = mindspore.Tensor(x_np)
grad = P.OnesLike()(x)

weight = Tensor(np.ones(2).astype(np.float32))
bias = Tensor(np.ones(2).astype(np.float32))
moving_mean = Tensor(np.ones(2).astype(np.float32))
moving_var_init = Tensor(np.ones(2).astype(np.float32))

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
bn_net = Batchnorm_Net(2, weight, bias, moving_mean, moving_var_init)
bn_net.set_train()
bn_grad = Grad(bn_net)
output = bn_grad(Tensor(x), Tensor(grad))

