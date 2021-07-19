import torch
import torch.nn.functional as F


def stable_softmax(q, temperature=1.0, dim=-1):
    v = q.max(dim=dim)[0].unsqueeze(dim)
    return F.softmax((q - v) / temperature, dim=dim)


def stable_log_softmax(q, temperature=1.0, dim=-1):
    v = q.max(dim=dim)[0].unsqueeze(dim)
    q_prime = q - v
    log_policy = q_prime - temperature * torch.log(
        torch.exp(q_prime / temperature).sum(dim=dim, keepdim=True)
    )
    return log_policy
