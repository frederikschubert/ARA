from collections import deque
from rlpyt.utils.logging import logger
from typing import List

import torch
from rlpyt.algos.qpg.sac import SAC
from rlpyt.algos.utils import quantile_regression_loss, valid_from_done
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.tensor import valid_mean


class DSAC(SAC):
    def __init__(self, risk_transform_fields: List[str] = ["q_pi"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_transform_fields = risk_transform_fields
        self.last_grads = list()

    def get_tau(self, samples, description):
        return self.agent.get_tau(
            self.batch_size, transform=description in self.risk_transform_fields
        )

    def compute_log_grad_variance(self):
        with torch.no_grad():
            grads = [
                torch.flatten(p.grad)
                for p in self.agent.pi_parameters()
                if p.grad is not None
            ]
            if grads:
                self.last_grads.append(torch.cat(grads))
            if len(self.last_grads) > 100:
                last_grads = torch.stack(self.last_grads)
                last_grads_var, last_grads_mean = torch.var_mean(last_grads, dim=0)
                logger.record_tabular(
                    "PiLogGradVar",
                    torch.log(last_grads_var.mean()).detach().cpu().numpy(),
                )
                self.last_grads.clear()

    def loss(self, samples):
        self.compute_log_grad_variance()

        if self.mid_batch_reset and not self.agent.recurrent:
            valid = torch.ones_like(samples.done, dtype=torch.float)  # or None
        else:
            valid = valid_from_done(samples.done)
        if self.bootstrap_timelimit:
            valid *= 1 - samples.timeout_n.float()
        agent_inputs, target_inputs, action, valid, returns, dones = buffer_to(
            (
                samples.agent_inputs,
                samples.target_inputs,
                samples.action,
                valid,
                samples.return_,
                samples.done_n,
            ),
            self.agent.device,
        )

        with torch.no_grad():
            target_action, target_log_pi, _ = self.agent.pi(*target_inputs)
            _, next_tau_hat, next_presum_tau = self.get_tau(samples, "q_target")
            target_q1, target_q2 = self.agent.target_q(
                *target_inputs, target_action, tau=next_tau_hat
            )
        min_target_q = torch.min(target_q1, target_q2)
        target_value = min_target_q - self._alpha * target_log_pi.unsqueeze(-1)
        disc = self.discount ** self.n_step_return
        y = (
            self.reward_scale * returns.reshape(self.batch_size, 1)
            + (1 - dones.float()).reshape(self.batch_size, 1) * disc * target_value
        )

        _, tau_hat, _ = self.get_tau(samples, "q_action")
        q1, q2 = self.agent.q(*agent_inputs, action, tau=tau_hat)

        q1_loss = valid_mean(
            quantile_regression_loss(q1, y, tau_hat, next_presum_tau)[0], valid
        )
        q2_loss = valid_mean(
            quantile_regression_loss(q2, y, tau_hat, next_presum_tau)[0], valid
        )

        new_action, log_pi, (pi_mean, pi_log_std) = self.agent.pi(*agent_inputs)
        with torch.no_grad():
            _, new_tau_hat, new_presum_tau = self.get_tau(samples, "q_pi")
        log_target1, log_target2 = self.agent.q(
            *agent_inputs, new_action, tau=new_tau_hat
        )

        log_target1 = torch.sum(new_presum_tau * log_target1, dim=1, keepdim=True)
        log_target2 = torch.sum(new_presum_tau * log_target2, dim=1, keepdim=True)

        min_log_target = torch.min(log_target1, log_target2).squeeze()
        prior_log_pi = self.get_action_prior(new_action.cpu())

        pi_losses = self._alpha * log_pi - min_log_target - prior_log_pi
        pi_loss = valid_mean(pi_losses, valid)

        if self.target_entropy is not None and self.fixed_alpha is None:
            alpha_losses = -self._log_alpha * (log_pi.detach() + self.target_entropy)
            alpha_loss = valid_mean(alpha_losses, valid)
        else:
            alpha_loss = None

        losses = (q1_loss, q2_loss, pi_loss, alpha_loss)
        values = tuple(val.detach() for val in (q1, q2, pi_mean, pi_log_std))
        return losses, values
