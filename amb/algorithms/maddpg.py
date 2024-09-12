import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from amb.agents.ddpg_agent import DDPGAgent
from amb.models.critic.q_critic import QCritic
from amb.utils.env_utils import check
from amb.utils.model_utils import update_linear_schedule, get_grad_norm
from amb.utils.trans_utils import _t2n

class MADDPG:
    def __init__(self, args, num_agents, obs_spaces, share_obs_space, act_spaces, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents
        self.action_type = act_spaces[0].__class__.__name__
        self.share_param = args['share_param']

        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.hidden_sizes = args["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = args["recurrent_n"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        self.batch_size = args["batch_size"]
        self.gamma = args["gamma"]
        self.lr = args["lr"]
        self.critic_lr = args["critic_lr"]
        self.expl_noise = args["expl_noise"]
        self.polyak = args["polyak"]
        self.use_policy_active_masks = args["use_policy_active_masks"]

        self.agents = []
        self.actors = []
        self.target_actors = []
        self.actor_optimizers = []

        if self.share_param:
            agent = DDPGAgent(args, obs_spaces[0], act_spaces[0], device)
            actor = agent.actor
            target_actor = deepcopy(actor)
            for p in target_actor.parameters():
                p.requires_grad = False
            optimizer = torch.optim.Adam(actor.parameters(), lr=self.lr)
            for agent_id in range(self.num_agents):
                self.agents.append(agent)
                self.actors.append(actor)
                self.target_actors.append(target_actor)
                self.actor_optimizers.append(optimizer)
        else:
            for agent_id in range(self.num_agents):
                agent = DDPGAgent(args, obs_spaces[agent_id], act_spaces[agent_id], device)
                actor = agent.actor
                target_actor = deepcopy(actor)
                for p in target_actor.parameters():
                    p.requires_grad = False
                optimizer = torch.optim.Adam(actor.parameters(), lr=self.lr)
                self.agents.append(agent)
                self.actors.append(actor)
                self.target_actors.append(target_actor)
                self.actor_optimizers.append(optimizer)
            
        self.critic = QCritic(args, share_obs_space, act_spaces, device)
        self.target_critic = deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        for agent_id in range(self.num_agents):
            self.actor_off_grad(agent_id)
        self.critic_off_grad()

    @staticmethod
    def create_agent(args, obs_space, act_space, device=torch.device("cpu")):
        return DDPGAgent(args, obs_space, act_space, device)

    def lr_decay(self, step, steps):
        if self.share_param:
            update_linear_schedule(self.actor_optimizers[0], step, steps, self.lr)
        else:
            for agent_id in range(self.num_agents):
                update_linear_schedule(self.actor_optimizers[agent_id], step, steps, self.lr)

    def update_critic(self, sample):
        obs = sample["obs"]
        share_obs = sample["share_obs"]
        actions = sample["actions_onehot"]
        rewards = sample["rewards"]
        gammas = sample["gammas"]
        next_obs = sample["next_obs"]
        next_share_obs = sample["next_share_obs"]
        rnn_states_actor = sample["rnn_states_actor"]
        target_rnn_states_actor = rnn_states_actor.copy()
        rnn_states_critic = sample["rnn_states_critic"]
        target_rnn_states_critic = rnn_states_critic.copy()
        masks = sample["masks"]
        active_masks = sample["active_masks"]
        dones_env = sample["dones_env"]
        filled = sample["filled"]
        available_actions = None
        next_available_actions = None
        if "available_actions" in sample:
            available_actions = sample["available_actions"]
            next_available_actions = sample["next_available_actions"]

        rewards = check(rewards).to(**self.tpdv)
        gammas = check(gammas).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        dones_env = check(dones_env).to(**self.tpdv)
        active_masks = check(active_masks).to(**self.tpdv)
        filled = check(filled).to(**self.tpdv).reshape(-1, 1, 1)
        filled = filled.expand_as(active_masks)

        # get hidden states for RNN
        if self.use_recurrent_policy:
            batch_size = rnn_states_actor.shape[0]
            target_rnn_states_actor = []
            target_rnn_states_critic = []
            for agent_id in range(self.num_agents):
                _, target_state_actor = self.target_actors[agent_id](
                    obs[:batch_size, agent_id], rnn_states_actor[:, agent_id], masks[:batch_size, agent_id], 
                    available_actions[:batch_size, agent_id] if available_actions is not None else None)
                _, target_state_critic = self.target_critic(
                    share_obs[:batch_size, agent_id], actions[:batch_size], 
                    rnn_states_critic[:, agent_id], masks[:batch_size, agent_id])
                target_rnn_states_actor.append(_t2n(target_state_actor))
                target_rnn_states_critic.append(_t2n(target_state_critic))
            target_rnn_states_actor = np.stack(target_rnn_states_actor, axis=1)
            target_rnn_states_critic = np.stack(target_rnn_states_critic, axis=1)

        # train critic
        self.critic_on_grad()

        next_actions = []
        for agent_id in range(self.num_agents):
            next_actions_dist, _ = self.target_actors[agent_id](
                next_obs[:, agent_id], target_rnn_states_actor[:, agent_id], masks[:, agent_id], 
                next_available_actions[:, agent_id] if available_actions is not None else None)
            next_actions.append(next_actions_dist.mode)
        next_actions = torch.stack(next_actions, dim=1)

        next_q_values = []
        for agent_id in range(self.num_agents):
            q_out, _ = self.target_critic(next_share_obs[:, agent_id], next_actions,
                                          target_rnn_states_critic[:, agent_id], masks[:, agent_id])
            next_q_values.append(q_out)
        next_q_values = torch.stack(next_q_values, dim=1)

        q_values = []
        for agent_id in range(self.num_agents):
            q_out, _ = self.critic(share_obs[:, agent_id], actions, 
                                   rnn_states_critic[:, agent_id], masks[:, agent_id])
            q_values.append(q_out)
        q_values = torch.stack(q_values, dim=1)
            
        q_targets = rewards + (self.gamma ** gammas) * next_q_values * (1 - dones_env)

        # print(torch.stack([q_values[:, 0, 0], q_targets[:, 0, 0]], dim=1))
        critic_loss = (q_values - q_targets) ** 2 

        if self.use_policy_active_masks:
            critic_loss = torch.sum(critic_loss * active_masks) / active_masks.sum()
        else:
            critic_loss = torch.sum(critic_loss * filled) / filled.sum()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm).item()
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()

        self.critic_off_grad()

        # print(critic_loss.item())

        return critic_loss, critic_grad_norm, q_targets, q_values

    def update_actor(self, sample):
        # [timestep*batch_size, num_agents, vshape]            
        obs = sample["obs"]
        share_obs = sample["share_obs"]
        actions = sample["actions_onehot"]
        rnn_states_actor = sample["rnn_states_actor"]
        rnn_states_critic = sample["rnn_states_critic"]
        masks = sample["masks"]
        active_masks = sample["active_masks"]
        filled = sample["filled"]
        available_actions = None
        if "available_actions" in sample:
            available_actions = sample["available_actions"]

        actions = check(actions).to(**self.tpdv)
        active_masks = check(active_masks).to(**self.tpdv)
        filled = check(filled).to(**self.tpdv).reshape(-1, 1, 1)
        filled = filled.expand_as(active_masks)

        actor_losses = []
        actor_grad_norms = []
        for agent_id in range(self.num_agents):
            all_actions = actions.detach().clone()
            self.actor_on_grad(agent_id)
            
            action_dist, _ = self.actors[agent_id](
                obs[:, agent_id], rnn_states_actor[:, agent_id], masks[:, agent_id],
                available_actions[:, agent_id] if available_actions is not None else None)
            if self.action_type == "Discrete":
                all_actions[:, agent_id] = F.gumbel_softmax(action_dist.logits, hard=True)
            elif self.action_type == "Box":
                all_actions[:, agent_id] = action_dist.mean
            value_pred, _ = self.critic(share_obs[:, agent_id], all_actions, 
                                     rnn_states_critic[:, agent_id], masks[:, agent_id])
            if self.use_policy_active_masks:
                actor_loss = - torch.sum(value_pred * active_masks[:, agent_id]) / active_masks[:, agent_id].sum()
            else:
                actor_loss = - torch.sum(value_pred * filled[:, agent_id]) / filled[:, agent_id].sum()
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()

            if self.use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm).item()
            else:
                actor_grad_norm = get_grad_norm(self.actors[agent_id].parameters())

            self.actor_optimizers[agent_id].step()
            self.actor_off_grad(agent_id)
            actor_losses.append(actor_loss)
            actor_grad_norms.append(actor_grad_norm)

        return actor_losses, actor_grad_norms


    def train(self, buffer):
        critic_train_info = {}

        critic_train_info["critic_loss"] = 0
        critic_train_info["critic_grad_norm"] = 0
        critic_train_info["q_targets"] = 0
        critic_train_info["q_values"] = 0

        actor_train_infos = [{
            "actor_loss": 0,
            "actor_grad_norm": 0
        } for _ in range(self.num_agents)]

        if self.use_recurrent_policy:
            data_generator = buffer.episode_generator(1, self.batch_size)
        else:
            data_generator = buffer.step_generator(1, self.batch_size)

        for sample in data_generator:
            critic_loss, critic_grad_norm, q_targets, q_values = self.update_critic(sample)
            actor_losses, actor_grad_norms = self.update_actor(sample)

            for agent_id in range(self.num_agents):
                self.soft_update(self.actors[agent_id], self.target_actors[agent_id])
            self.soft_update(self.critic, self.target_critic)

            critic_train_info["critic_loss"] += critic_loss.item()        
            critic_train_info["critic_grad_norm"] += critic_grad_norm
            critic_train_info["q_targets"] += q_targets.mean().item()
            critic_train_info["q_values"] += q_values.mean().item()

            for agent_id in range(self.num_agents):
                actor_train_infos[agent_id]["actor_loss"] += actor_losses[agent_id].item()
                actor_train_infos[agent_id]["actor_grad_norm"] += actor_grad_norms[agent_id]

        return actor_train_infos, critic_train_info

    def prep_training(self):
        for actor in self.actors:
            actor.train()
        self.critic.train()

    def prep_rollout(self):
        for actor in self.actors:
            actor.eval()
        self.critic.eval()

    def critic_on_grad(self):
        """Turn on the gradient for the critic."""
        for param in self.critic.parameters():
            param.requires_grad = True

    def critic_off_grad(self):
        """Turn off the gradient for the critic."""
        for param in self.critic.parameters():
            param.requires_grad = False

    def actor_on_grad(self, id):
        """Turn on the gradient for the actors."""
        for param in self.actors[id].parameters():
            param.requires_grad = True

    def actor_off_grad(self, id):
        """Turn off the gradient for the actors."""
        for param in self.actors[id].parameters():
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
