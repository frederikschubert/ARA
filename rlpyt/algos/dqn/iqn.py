from typing import List, Tuple, cast

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.algos.utils import quantile_regression_loss
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


def plot_quantiles(quantiles: torch.Tensor, obs: torch.Tensor):
    fig = plt.figure(constrained_layout=True)
    layout = """
    AA
    BB
    """
    ax_dict = fig.subplot_mosaic(layout)
    for a in range(quantiles.shape[-1]):
        sns.ecdfplot(
            quantiles[:, a].cpu().detach().numpy(), ax=ax_dict["A"], label=str(a)
        )
    if obs.shape[0] != 3:
        obs_img = to_pil_image(
            make_grid(obs.unsqueeze(1), nrow=obs.shape[0]).cpu().detach()
        )
    else:
        obs_img = to_pil_image(obs.cpu().detach())
    ax_dict["B"].imshow(obs_img)
    ax_dict["B"].axis("off")
    ax_dict["A"].legend()
    plt.close(fig)
    return fig


class IQN(DQN):
    def __init__(
        self, kappa: float = 1.0, risk_transform_fields: List[str] = [], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.kappa = kappa
        self.risk_transform_fields = risk_transform_fields

    def get_tau(self, samples, description: str):
        return self.agent.get_tau(
            self.batch_size, transform=description in self.risk_transform_fields
        )

    def loss(self, samples):
        obs, actions, rewards, dones, next_obs = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            buffer_to(
                (
                    samples.agent_inputs.observation,
                    samples.action,
                    samples.return_,
                    samples.done_n,
                    samples.target_inputs.observation,
                ),
                device=self.agent.device,
            ),
        )
        # shape: (batch_size, num_quantiles)
        tau = self.get_tau(samples, "q")
        num_quantiles = tau.shape[-1]

        quantiles = self.agent.model(obs, tau=tau)
        quantile_values = torch.gather(
            quantiles,
            dim=-1,
            index=actions.reshape(self.batch_size, 1, 1).expand(
                self.batch_size, num_quantiles, 1
            ),
        ).squeeze()

        if self.update_counter % 100 == 0:
            fig = plot_quantiles(quantiles[0], obs[0])
            logger.record_image("quantiles", fig)

        with torch.no_grad():
            tau_next = self.get_tau(samples, "q_target_action")
            next_quantile_values = self.agent.model(next_obs, tau=tau_next)
            next_q = next_quantile_values.mean(dim=-2)
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)

            tau_next = self.get_tau(samples, "q_target")
            if self.double_dqn:
                next_quantiles = self.agent.target_model(next_obs, tau=tau_next)
            else:
                next_quantiles = self.agent.model(next_obs, tau=tau_next)
            next_quantile_values = torch.gather(
                next_quantiles,
                dim=-1,
                index=next_actions.reshape(self.batch_size, 1, 1).expand(
                    self.batch_size, num_quantiles, 1
                ),
            ).squeeze()

            target_quantile_values = (
                rewards.reshape(self.batch_size, 1)
                + (1.0 - dones.reshape(self.batch_size, 1).float())
                * self.discount ** self.n_step_return
                * next_quantile_values
            )

        quantile_huber_loss, td_errors = quantile_regression_loss(
            quantile_values, target_quantile_values, tau, kappa=self.kappa
        )

        return (
            quantile_huber_loss,
            td_errors.mean(dim=(-2, -1)).detach().abs().cpu(),
        )
