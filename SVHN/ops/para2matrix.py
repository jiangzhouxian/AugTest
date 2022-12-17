import torch
from torch.autograd import Function


class Theta2Matrix(Function):

    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        M = torch.tensor(
            [[torch.cos(theta), -torch.sin(theta), 0],
             [torch.sin(theta), torch.cos(theta), 0]]
        )
        return M

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        dM_dtheta = torch.tensor(
            [[-torch.sin(theta), -torch.cos(theta), 0],
             [torch.cos(theta), -torch.sin(theta), 0]]
        )
        out = grad_output * dM_dtheta
        out = out.sum()
        out = torch.tensor([out])
        return out
theta2matrix = Theta2Matrix.apply

class Tx2Matrix(Function):

    @staticmethod
    def forward(ctx, tx):
        ctx.save_for_backward(tx)
        M = torch.tensor(
            [[1,0,tx],
             [0,1,0]]
        )
        return M

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        dM_dtx = torch.tensor(
            [[0, 0, 1],
             [0, 0, 0]]
        )
        out = grad_output * dM_dtx
        out = out.sum()
        out = torch.tensor([out])
        return out
tx2Matrix = Tx2Matrix.apply


class Sx2Matrix(Function):
    @staticmethod
    def forward(ctx,sx):
        ctx.save_for_backward(sx)
        M = torch.tensor(
            [[1, sx, 0], [0,1,0]]
        )
        return M

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        dM_dsx = torch.tensor(
            [[0, 1, 0],
             [0, 0, 0]]
        )
        out = grad_output * dM_dsx
        out = out.sum()
        out = torch.tensor([out])
        return out
sx2Matrix = Sx2Matrix.apply

class Zx2Matrix(Function):
    @staticmethod
    def forward(ctx, zx):
        ctx.save_for_backward(zx)
        M = torch.tensor(
            [[zx,0,0], [0,1,0]]
        )
        return M

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        dM_dzx = torch.tensor(
            [[1, 0, 0],
             [0, 0, 0]]
        )
        out = grad_output * dM_dzx
        out = out.sum()
        out = torch.tensor([out])
        return out
zx2Matrix = Zx2Matrix.apply