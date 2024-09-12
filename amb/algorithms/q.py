import os
from copy import deepcopy
import torch
import torch.nn as nn
from amb.agents.q_agent import QAgent
from amb.models.mixer.qmix import QMixer
from amb.models.mixer.vdn import VDNMixer
from amb.utils.env_utils import check
from amb.utils.model_utils import update_linear_schedule, get_grad_norm


class Q:
    def __init__(self, args, num_agents, obs_spaces, share_obs_space, act_spaces, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents
        self.action_type = act_spaces[0].__class__.__name__
        self.share_param = args["share_param"]

        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.hidden_sizes = args["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = args["recurrent_n"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        self.batch_size = args["batch_size"]
        self.gamma = args["gamma"]
        self.lr = args["lr"]
        self.polyak = args["polyak"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.mixer = args["mixer"]

        self.agents = []
        self.actors = []
        self.target_actors = []
        self.params = []

        if self.share_param:
            agent = QAgent(args, obs_spaces[0], act_spaces[0], device)
            actor = agent.actor
            target_actor = deepcopy(actor)
            for p in target_actor.parameters():
                p.requires_grad = False
            self.params += list(actor.parameters())
            for agent_id in range(self.num_agents):
                self.agents.append(agent)
                self.actors.append(actor)
                self.target_actors.append(target_actor)
        else:
            for agent_id in range(self.num_agents):
                agent = QAgent(
                    args, obs_spaces[agent_id], act_spaces[agent_id], device
                )
                actor = agent.actor
                target_actor = deepcopy(actor)
                for p in target_actor.parameters():
                    p.requires_grad = False
                self.params += list(actor.parameters())
                self.agents.append(agent)
                self.actors.append(actor)
                self.target_actors.append(target_actor)

        if self.mixer is not None:
            if self.mixer == "qmix":
                self.critic = QMixer(args, num_agents, share_obs_space, device)
            elif self.mixer == "vdn":
                self.critic = VDNMixer()
            else:
                raise ValueError("Mixer {} not recognised.".format(self.mixer))
            self.target_critic = deepcopy(self.critic)
            self.params += list(self.critic.parameters())
        else:
            self.critic = PlaceholderCritic()
            self.target_critic = deepcopy(self.critic)

        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

        self._off_grad()
           

    @staticmethod
    def create_agent(args, obs_space, act_space, device=torch.device("cpu")):
        return QAgent(args, obs_space, act_space, device)

    def lr_decay(self, step, steps):
        update_linear_schedule(self.optimizer, step, steps, self.lr)

    def update(self, sample):
        # [timestep*batch_size, num_agents, vshape]
        obs = sample["obs"]
        share_obs = sample["share_obs"]
        actions = sample["actions"]
        rnn_states_actor = sample["rnn_states_actor"]
        next_obs = sample["next_obs"]
        next_share_obs = sample["next_share_obs"]
        masks = sample["masks"]
        rewards = sample["rewards"]
        gammas = sample["gammas"]
        active_masks = sample["active_masks"]
        dones_env = sample["dones_env"]
        filled = sample["filled"]
        available_actions = None
        if "available_actions" in sample:
            available_actions = sample["available_actions"]

        actions = check(actions).to(**self.tpdv)
        active_masks = check(active_masks).to(**self.tpdv)
        dones_env = check(dones_env).to(**self.tpdv)
        share_obs = check(share_obs).to(**self.tpdv)
        rewards = check(rewards).to(**self.tpdv)
        gammas = check(gammas).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        filled = check(filled).to(**self.tpdv).reshape(-1, 1, 1)
        filled = filled.expand_as(active_masks)

        q_values = []
        next_q_values = []

        self._on_grad()
        # Calculate estimated Q-Values
        for agent_id in range(self.num_agents):
            # Calculate estimated Q-Values
            q_dist, _ = self.actors[agent_id](
                obs[:, agent_id],
                rnn_states_actor[:, agent_id],
                masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions is not None
                else None,
            )
            q_values.append(q_dist.logits)
        q_values = torch.stack(q_values, dim=1)

        # Pick the Q-Values for the actions taken by the agent
        q_values = torch.gather(q_values, dim=2, index=actions.long()) # [B, N, 1]

        # Calculate the Q-Values necessary for the target
        for agent_id in range(self.num_agents):
            # Calculate estimated Q-Values
            next_q_dist, _ = self.target_actors[agent_id](
                next_obs[:, agent_id],
                rnn_states_actor[:, agent_id].copy(),
                masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions is not None
                else None,
            )
            next_q_values.append(next_q_dist.logits)
        next_q_values = torch.stack(next_q_values, dim=1)

        # Max over target Q-Values
        next_q_values = next_q_values.max(dim=2, keepdim=True)[0] # [B, N, 1]

        # Mix the Q-Values, squeeze to [B, N, 1]
        if self.mixer is not None:
            mixed_q_values = []
            next_mixed_q_values = []
            for agent_id in range(self.num_agents):
                q_out = self.critic(q_values, share_obs[:, agent_id]).squeeze(-1)
                next_q_out = self.target_critic(next_q_values, next_share_obs[:, agent_id]).squeeze(-1)
                mixed_q_values.append(q_out)
                next_mixed_q_values.append(next_q_out)
            q_values = torch.stack(mixed_q_values, dim=1)
            next_q_values = torch.stack(next_mixed_q_values, dim=1)

        # Calculate n-step Q-Learning targets
        q_targets = rewards + (self.gamma ** gammas) * next_q_values * (1 - dones_env)

        # Calculate actor loss
        critic_loss = (q_values - q_targets.detach()) ** 2

        if self.use_policy_active_masks:
            critic_loss = torch.sum(critic_loss * active_masks) / active_masks.sum()
        else:
            critic_loss = torch.sum(critic_loss * filled) / filled.sum()

        self.optimizer.zero_grad()
        critic_loss.backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.params, self.max_grad_norm).item()
        else:
            critic_grad_norm = get_grad_norm(self.params)

        self.optimizer.step()
        self._off_grad()

        return critic_loss, critic_grad_norm, q_targets, q_values

    def train(self, buffer):
        critic_train_info = {}

        critic_train_info["critic_loss"] = 0
        critic_train_info["critic_grad_norm"] = 0
        critic_train_info["q_targets"] = 0
        critic_train_info["q_values"] = 0

        actor_train_infos = [
            {"actor_loss": 0} for _ in range(self.num_agents)
        ]

        if self.use_recurrent_policy:
            data_generator = buffer.episode_generator(1, self.batch_size)
        else:
            data_generator = buffer.step_generator(1, self.batch_size)

        for sample in data_generator:
            critic_loss, critic_grad_norm, q_targets, q_values = self.update(sample)

            for agent_id in range(self.num_agents):
                self.soft_update(self.actors[agent_id], self.target_actors[agent_id])
                self.soft_update(self.critic, self.target_critic)

            critic_train_info["critic_loss"] += critic_loss.item()        
            critic_train_info["critic_grad_norm"] += critic_grad_norm
            critic_train_info["q_targets"] += q_targets.mean().item()
            critic_train_info["q_values"] += q_values.mean().item()

        return actor_train_infos, critic_train_info

    def prep_training(self):
        for actor in self.actors:
            actor.train()
        self.critic.train()

    def prep_rollout(self):
        for actor in self.actors:
            actor.eval()
        self.critic.eval()

    def _on_grad(self):
        """Turn on the gradient for the actors."""
        for agent_id in range(self.num_agents):
            for param in self.actors[agent_id].parameters():
                param.requires_grad = True
        for param in self.critic.parameters():
            param.requires_grad = True

    def _off_grad(self):
        """Turn off the gradient for the actors."""
        for agent_id in range(self.num_agents):
            for param in self.actors[agent_id].parameters():
                param.requires_grad = False
        for param in self.critic.parameters():
            param.requires_grad = False

    def soft_update(self, model: nn.Module, target_model: nn.Module):
        for param_target, param in zip(target_model.parameters(), model.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.polyak) + param.data * self.polyak)

    def save(self, path):
        if self.share_param:
            self.agents[0].save(path)
        else:
            for agent_id in range(self.num_agents):
                self.agents[agent_id].save(os.path.join(path, str(agent_id)))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))

    def restore(self, path):
        if self.share_param:
            self.agents[0].restore(path)
        else:
            for agent_id in range(self.num_agents):
                self.agents[agent_id].restore(os.path.join(path, str(agent_id)))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth")))


class PlaceholderCritic(nn.Module):
    pass