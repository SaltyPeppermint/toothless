from gymnasium import Env
import torch
from torch import DeviceObjType, nn
from torch.distributions.categorical import Categorical
from torch_geometric.data import Data


from model import SketchEmbed
from rl_env import Observation


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.observations[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class SketchAgent(nn.Module):
    def __init__(self, env: Env, embed_dim: int = 256, hidden_dim: int = 256):
        super(SketchAgent, self).__init__()
        self.actor = nn.Sequential(
            SketchEmbed(
                env.observation_space,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
            ),
            nn.Linear(hidden_dim, env.action_space.n),
        )
        self.critic = nn.Sequential(
            SketchEmbed(
                env.observation_space,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
            ),
            nn.Linear(hidden_dim, 1),
        )

    def evaluate(self, observation, action):
        action_probs = self.actor(observation)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(observation)

        return action_logprobs, state_values, dist_entropy

    def act(self, observation):
        action_probs = self.actor(observation)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(observation)

        return action.detach(), action_logprob.detach(), state_val.detach()

    # def to(self, device):
    #     self.critic = self.critic.to(device)
    #     self.actor = self.actor.to(device)


class PPO:
    def __init__(
        self,
        device,
        env,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
    ):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = SketchAgent(env, embed_dim=256, hidden_dim=256).to(device)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = SketchAgent(env, embed_dim=256, hidden_dim=256).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, observation: Observation):
        with torch.no_grad():
            observation.to(self.device)
            action, action_logprob, state_val = self.policy_old.act(observation)

        self.buffer.observations.append(observation)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_observations = (
            torch.squeeze(torch.stack(self.buffer.observations, dim=0))
            .detach()
            .to(self.device)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
            .detach()
            .to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
            .detach()
            .to(self.device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(self.device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_observations, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )


def dict_device_send(observations: dict[str, Data], device: DeviceObjType):
    observations["lhs"].to(device)
    observations["rhs"].to(device)
    observations["sketch"].to(device)
