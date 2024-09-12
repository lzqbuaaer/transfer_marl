import os
from copy import deepcopy
import torch
import torch.nn as nn
from amb.agents.q_agent import QAgent
from amb.models.mixer.qtran import QTranBase
from amb.utils.env_utils import check, get_onehot_shape_from_act_space
from amb.utils.model_utils import update_linear_schedule, get_grad_norm


class QTran:
    def __init__(self, args, num_agents, obs_spaces, share_obs_spaces, act_spaces, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents
        self.num_actions = get_onehot_shape_from_act_space(act_spaces[0])
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
        self.optim_alpha = args["optim_alpha"]
        self.optim_eps = args["optim_eps"]
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
            if self.mixer == "qtran_base":
                self.critic = QTranBase(args, num_agents, share_obs_spaces, act_spaces, device)
            elif self.mixer == "qtran_alt":
                raise Exception("Not implemented here!")
            self.params += list(self.critic.parameters())
        else:
            self.critic = PlaceholderCritic()
        self.target_critic = deepcopy(self.critic)

        self.optimizer = torch.optim.RMSprop(params=self.params, lr=self.lr, alpha=self.optim_alpha, eps=self.optim_eps)

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
        actions_onehot = sample["actions_onehot"]
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
        actions_onehot = check(actions_onehot).to(**self.tpdv)
        active_masks = check(active_masks).to(**self.tpdv)
        dones_env = check(dones_env).to(**self.tpdv)
        share_obs = check(share_obs).to(**self.tpdv)
        rewards = check(rewards).to(**self.tpdv)
        gammas = check(gammas).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        filled = check(filled).to(**self.tpdv).reshape(-1, 1, 1)
        filled = filled.expand_as(active_masks)

        q_values = []
        hidden_status = []
        target_q_values = []
        target_hidden_status = []

        self._on_grad()
        # Calculate estimated Q-Values
        for agent_id in range(self.num_agents):
            # Calculate estimated Q-Values
            q_dist, rnn_status = self.actors[agent_id](
                obs[:, agent_id],
                rnn_states_actor[:, agent_id],
                masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions is not None
                else None,
            )
            q_values.append(q_dist.logits)
            hidden_status.append(rnn_status)
        q_values = torch.stack(q_values, dim=1)
        hidden_status = torch.stack(hidden_status, dim=1)

        # Pick the Q-Values for the actions taken by the agent
        chosen_action_qvals = torch.gather(q_values, dim=2, index=actions.long()) # [B, N, 1]

        # Calculate the Q-Values necessary for the target
        for agent_id in range(self.num_agents):
            # Calculate estimated Q-Values
            target_q_dist, target_rnn_status = self.target_actors[agent_id](
                next_obs[:, agent_id],
                rnn_states_actor[:, agent_id].copy(),
                masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions is not None
                else None,
            )
            target_q_values.append(target_q_dist.logits)
            target_hidden_status.append(target_rnn_status)
        target_q_values = torch.stack(target_q_values, dim=1)
        target_hidden_status = torch.stack(target_hidden_status, dim=1)

        # Best joint action computed by target agents
        target_max_actions = target_q_values.max(dim=2, keepdim=True)[1] # [B, N, 1]
        # Best joint-action computed by regular agents
        mac_out_maxs = q_values.clone()
        max_actions_qvals, _ = mac_out_maxs.max(dim=2, keepdim=True)

        # Mix the Q-Values, squeeze to [B, N, 1]
        if self.mixer is not None:
            if self.mixer == "qtran_base":
                # -- TD Loss --
                # Joint-action Q-Value estimates
                joint_qs, vs = self.critic(sample, hidden_status) # [B, N, 1]

                # Need to argmax across the target agents' actions to compute target joint-action Q-Values
                max_actions = torch.zeros_like(actions_onehot).to(self.device)
                max_actions_onehot = max_actions.scatter(-1, target_max_actions.long(), 1)
                target_joint_qs, target_vs = self.target_critic(sample, hidden_states=target_hidden_status, actions=max_actions_onehot)

                # Td loss targets
                td_targets = rewards + (self.gamma ** gammas) * target_joint_qs * (1 - dones_env)
                td_error = (joint_qs - td_targets.detach())
                mask = active_masks if self.use_policy_active_masks else filled
                td_loss = torch.sum(td_error ** 2 * mask) / mask.sum()
                # -- TD Loss End --

                # -- Opt Loss --
                # Argmax across the current agents' actions
                max_actions_current_ = torch.zeros_like(actions_onehot).to(self.device)
                max_actions_current_onehot = max_actions_current_.scatter(-1, target_max_actions.long(), 1)
                max_joint_qs, _ = self.critic(sample, hidden_status, actions=max_actions_current_onehot)

                opt_error = max_actions_qvals - max_joint_qs.detach() + vs
                opt_loss = torch.sum(opt_error ** 2 * mask) / mask.sum()
                # -- Opt Loss End --

                # -- Nopt Loss --
                nopt_values = chosen_action_qvals - joint_qs.detach() + vs
                nopt_error = nopt_values.clamp(max=0)
                nopt_loss = torch.sum(nopt_error ** 2 * mask) / mask.sum()
                # -- Nopt Loss End --

        elif self.args.mixer == "qtran_alt":
            raise Exception("Not supported yet.")

        # loss
        critic_loss = td_loss + self.args["opt_loss"] * opt_loss + self.args["nopt_min_loss"] * nopt_loss

        # Optimise
        self.optimizer.zero_grad()
        critic_loss.backward()
        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.params, self.max_grad_norm).item()
        else:
            critic_grad_norm = get_grad_norm(self.params)
        self.optimizer.step()
        self._off_grad()

        return critic_loss, td_loss, opt_loss, nopt_loss, critic_grad_norm, target_q_values, q_values

    def train(self, buffer):
        critic_train_info = {}

        critic_train_info["critic_loss"] = 0
        critic_train_info["td_loss"] = 0
        critic_train_info["opt_loss"] = 0
        critic_train_info["nopt_loss"] = 0
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
            critic_loss, td_loss, opt_loss, nopt_loss, critic_grad_norm, q_targets, q_values = self.update(sample)

            for agent_id in range(self.num_agents):
                self.soft_update(self.actors[agent_id], self.target_actors[agent_id])
                self.soft_update(self.critic, self.target_critic)

            critic_train_info["critic_loss"] += critic_loss.item()        
            critic_train_info["td_loss"] += td_loss.item()        
            critic_train_info["opt_loss"] += opt_loss.item()        
            critic_train_info["nopt_loss"] += nopt_loss.item()        
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