import torch
from torch.optim.optimizer import Optimizer
import math

class RefinedSFAdamW(Optimizer):
    """
    改进版 Refined SF-AdamW (加入解耦参数 C)
    """
    def __init__(self, params, lr=1e-3, betas=(0.95, 0.99), eps=1e-8, weight_decay=0.0, warmup_steps=0, c_factor=200.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup_steps=warmup_steps, c_factor=c_factor)
        super().__init__(params, defaults)
        self.step_idx = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_idx += 1
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            warmup_steps = group['warmup_steps']
            weight_decay = group['weight_decay']
            eps = group['eps']
            c_factor = group['c_factor'] # 提取你们的核心参数 C

            if warmup_steps > 0:
                step_size = lr * min(1.0, self.step_idx / warmup_steps)
            else:
                step_size = lr

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['z'] = p.clone()
                    state['x'] = p.clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)

                z, x, exp_avg_sq = state['z'], state['x'], state['exp_avg_sq']

                # 核心改进：引入 C 参数进行解耦
                adjusted_beta2 = beta2 * (1.0 - math.exp(-c_factor / max(1, self.step_idx)))
                
                exp_avg_sq.mul_(adjusted_beta2).addcmul_(grad, grad, value=1 - adjusted_beta2)
                denom = exp_avg_sq.sqrt().add_(eps)

                if weight_decay > 0:
                    z.mul_(1 - lr * weight_decay)

                z.addcdiv_(grad, denom, value=-step_size)
                
                # 动量平滑受到 C 的动态约束
                dynamic_beta1 = beta1 * (c_factor / (c_factor + 1)) 
                x.lerp_(z, 1 - dynamic_beta1)

                p.copy_(z).lerp_(x, dynamic_beta1)

        return loss