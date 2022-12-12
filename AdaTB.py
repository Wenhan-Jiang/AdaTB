import math
import torch
from torch.optim import Optimizer


class AdaTB(Optimizer):
    """Implements AdaTB algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-4)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amstbound (boolean, optional): whether to use the AMSTBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        An adaptive gradient method with transformation bound
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-4,
                 eps=1e-8, weight_decay=0, amstbound=False, clip=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amstbound=amstbound, clip=clip)
        super(AdaTB, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaTB, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amstbound', False)
            group.setdefault('clip', False)
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amstbound = group['amstbound']
                clip = group['clip']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amstbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_rt'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amstbound:
                    max_rt = state['max_rt']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = exp_avg_sq / bias_correction2
                step = group['gamma'] * state['step']

                relu = ((math.exp(step) - math.exp(-step)) / ((math.exp(step) + math.exp(-step))))
                relu = (1 - relu ** 2) * (1 - denom) + denom
                rt = denom / relu  #when rt=denom Adam
                if amstbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_rt, rt, out=max_rt)
                    # Use the max. for normalizing running avg. of gradient
                    rt = max_rt.sqrt().add_(group['eps'])
                else:
                    rt = rt.sqrt().add_(group['eps'])

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                step_size = group['lr'] / bias_correction1
                step_size = torch.full_like(rt, step_size)
                if clip:
                    final_lr = group['final_lr'] * group['lr'] / base_lr
                    lower_bound = final_lr * ((math.exp(step)-math.exp(-step)) / (math.exp(step) + math.exp(-step)))
                    upper_bound = (1 - (math.exp(step)-math.exp(-step)) / (math.exp(step) + math.exp(-step)) ** 2) * abs(1 - final_lr) + final_lr
                    step_size.div_(rt).clamp_(lower_bound, upper_bound).mul_(exp_avg)
                    p.data.add_(-step_size)
                else:
                    step_size.div_(rt).mul_(exp_avg)
                    p.data.add_(-step_size)

        return loss