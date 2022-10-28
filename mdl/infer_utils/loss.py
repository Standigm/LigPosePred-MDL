from torch.nn.modules import Module
from torch.nn import _reduction as _Reduction

from torch import Tensor
from typing import Callable, Optional

from torchvision.ops import sigmoid_focal_loss


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'maen') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class SigmoidFocalLoss(_Loss):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super(SigmoidFocalLoss, self).__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return sigmoid_focal_loss(input, target,
                                  alpha=self.alpha,
                                  gamma=self.gamma,
                                  reduction=self.reduction)


# Check implementation.
if __name__ == '__main__':
    import torch

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    size = 10
    pred = torch.randn(size)
    target = torch.randint(0, 2, (size,)).float()
    pred = pred.unsqueeze(-1).to(device)
    target = target.unsqueeze(-1).to(device)

    print(pred)
    print(target)

    criterion1 = SigmoidFocalLoss()
    criterion2 = torch.nn.BCEWithLogitsLoss()
    loss1a = criterion1(pred, target)
    loss1b = sigmoid_focal_loss(pred, target)
    loss2 = criterion2(pred, target)

    print(loss1a)
    print(loss1b)
    print(loss2)
