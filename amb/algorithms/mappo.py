import os
import torch
from torch import nn
import numpy as np
from amb.agents.ppo_agent import PPOAgent
from amb.models.critic.v_critic import VCritic
from amb.utils.model_utils import update_linear_schedule, get_grad_norm, huber_loss, mse_loss
from amb.utils.env_utils import check


class MAPPO:
    def __init__(self, args, num_agents, obs_spaces, share_obs_space, act_spaces, device=torch.device("cpu"), agent_type="ori_victim", ally_num=2):
        # save arguments
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents
        self.share_param = args['share_param']

        self.data_chunk_length = args["data_chunk_length"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.action_aggregation = args["action_aggregation"]

        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]

        self.entropy_coef = args["entropy_coef"]
        self.clip_param = args["clip_param"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        self.use_clipped_value_loss = args["use_clipped_value_loss"]
        self.value_loss_coef = args["value_loss_coef"]
        self.use_huber_loss = args["use_huber_loss"]
        self.huber_delta = args["huber_delta"]

        self.lr = args["lr"]
        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        
        self.env_belief = args.get("env_belief", False)
        if self.env_belief:
            env_prior_path = args.get("env_prior_path", "./env_prior.npy")
            self.env_prior = torch.tensor(np.load(env_prior_path)).to(device)
            self.belief_epoch = args["belief_epoch"]
            self.belief_num_mini_batch = args["belief_num_mini_batch"]

        self.obs_spaces = obs_spaces
        self.share_obs_space = share_obs_space
        self.act_spaces = act_spaces
        self.action_type = self.act_spaces[0].__class__.__name__
        
        self.agents = []
        self.actors = []
        self.actor_optimizers = []
        if self.env_belief: 
            self.belief_optimizers = []

        if self.share_param:
            agent = PPOAgent(args, obs_spaces[0], act_spaces[0], device=device, ally_num=ally_num, agent_type=agent_type)
            optimizer = torch.optim.Adam(
                agent.actor.parameters(),
                lr=self.lr,
                eps=self.opti_eps,
                weight_decay=self.weight_decay,
            )
            if self.env_belief:
                belief_optimizer = torch.optim.Adam(
                    agent.belief.parameters(),
                    lr=self.lr,
                    eps=self.opti_eps,
                    weight_decay=self.weight_decay
                )
            for agent_id in range(self.num_agents):
                self.agents.append(agent)
                self.actors.append(agent.actor)
                self.actor_optimizers.append(optimizer)
                if self.env_belief:    
                    self.belief_optimizers.append(belief_optimizer)
        else:
            for agent_id in range(self.num_agents):
                agent = PPOAgent(args, obs_spaces[agent_id], act_spaces[agent_id], device=device, ally_num=ally_num, agent_type=agent_type)
                optimizer = torch.optim.Adam(
                    agent.actor.parameters(),
                    lr=self.lr,
                    eps=self.opti_eps,
                    weight_decay=self.weight_decay,
                )
                self.agents.append(agent)
                self.actors.append(agent.actor)
                self.actor_optimizers.append(optimizer)
                if self.env_belief:
                    belief_optimizer = torch.optim.Adam(
                        agent.belief.parameters(),
                        lr=self.lr,
                        eps=self.opti_eps,
                        weight_decay=self.weight_decay
                    )
                    self.belief_optimizers.append(belief_optimizer)

        self.critic = VCritic(args, self.share_obs_space, device=self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    @staticmethod
    def create_agent(args, obs_space, act_space, device=torch.device("cpu"), ally_num=2, agent_type="ori_victim"):
        return PPOAgent(args, obs_space, act_space, device, ally_num=ally_num, agent_type=agent_type)

    def lr_decay(self, episode, episodes):
        if self.share_param:
            update_linear_schedule(self.actor_optimizers[0], episode, episodes, self.lr)
        else:
            for agent_id in range(self.num_agents):
                update_linear_schedule(self.actor_optimizers[agent_id], episode, episodes, self.lr)

    def evaluate_actions(self, agent_id, obs, rnn_states, action, masks, available_actions=None, active_masks=None, env_belief=None):
        action = check(action).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        action_dist, _ = self.actors[agent_id](obs, rnn_states, masks, available_actions, env_belief=env_belief)
        action_log_probs = action_dist.log_probs(action)
        if active_masks is not None:
            if self.action_type == "Discrete":
                dist_entropy = (
                    action_dist.entropy() * active_masks.squeeze(-1)
                ).sum() / active_masks.sum()
            else:
                dist_entropy = (
                    action_dist.entropy() * active_masks
                ).sum() / active_masks.sum()
        else:
            dist_entropy = action_dist.entropy().mean()

        return action_log_probs, dist_entropy

    def update_actor(self, sample, agent_id):
        obs = sample["obs"]
        rnn_states_actor = sample["rnn_states_actor"]
        actions = sample["actions"]
        masks = sample["masks"]
        active_masks = sample["active_masks"]
        old_action_log_probs = sample["action_log_probs"]
        target_advantage = sample["advantages"]
        available_actions = None
        if "available_actions" in sample:
            available_actions = sample["available_actions"]
        if self.env_belief:
            belief = sample["belief"]
        else:
            belief = None

        old_action_log_probs = check(old_action_log_probs).to(**self.tpdv)
        target_advantage = check(target_advantage).to(**self.tpdv)
        active_masks = check(active_masks).to(**self.tpdv)

        # reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy = self.evaluate_actions(
            agent_id, obs, rnn_states_actor, actions, masks, available_actions, active_masks, env_belief=belief)
        # update actor
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs), dim=-1, keepdim=True)

        surr1 = imp_weights * target_advantage
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * target_advantage

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks
            ).sum() / active_masks.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.actor_optimizers[agent_id].zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.actors[agent_id].parameters())

        self.actor_optimizers[agent_id].step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights
    
    def cal_value_loss(self, values, value_preds_batch, return_batch, value_normalizer=None):
        """Calculate value function loss.
        Args:
            values: (torch.Tensor) value function predictions.
            value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
            return_batch: (torch.Tensor) reward to go returns.
            value_normalizer: (PopArt) normalize the rewards, denormalize critic outputs.
        Returns:
            value_loss: (torch.Tensor) value function loss.
        """
        if value_normalizer is not None:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param
            )
            error_clipped = value_normalizer(return_batch) - value_pred_clipped
            error_original = value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param
            )
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def update_critic(self, sample, value_normalizer=None):
        share_obs = sample["share_obs"]
        rnn_states_critic = sample["rnn_states_critic"]
        value_preds = sample["value_preds"]
        returns = sample["returns"]
        masks = sample["masks"]

        value_preds = check(value_preds).to(**self.tpdv)
        returns = check(returns).to(**self.tpdv)

        values, _ = self.critic(share_obs, rnn_states_critic, masks)

        value_loss = self.cal_value_loss(
            values, value_preds, returns, value_normalizer=value_normalizer)

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm
    
    def update_belief(self, sample, agent_id):
        assert self.env_belief
        next_obs = sample["next_obs"]
        rewards = sample["rewards"]
        obs = sample["obs"]
        belief = sample["belief"]
        next_rnn_states_belief = sample["next_rnn_states_belief"]
        next_masks = sample["next_masks"]
        next_masks = check(next_masks).to(**self.tpdv)
        
        belief, _ = self.agents[agent_id].forward_belief(next_obs, rewards, obs, belief, next_rnn_states_belief, next_masks)
        difference = ((belief - self.env_prior) ** 2 * next_masks).sum()

        self.belief_optimizers[agent_id].zero_grad()

        difference.backward()

        if self.use_max_grad_norm:
            belief_grad_norm = nn.utils.clip_grad_norm_(self.agents[agent_id].belief.parameters(), self.max_grad_norm)
        else:
            belief_grad_norm = get_grad_norm(self.agents[agent_id].belief.parameters())

        self.belief_optimizers[agent_id].step()

        return difference, belief_grad_norm
    
    def train_belief(self, buffer, agent_id):
        train_info = {}
        train_info["belief_difference_loss"] = 0
        train_info["belief_grad_norm"] = 0

        for _ in range(self.belief_epoch):
            if self.use_recurrent_policy:
                data_generator = buffer.chunk_generator(self.belief_num_mini_batch, self.data_chunk_length)
            else:
                data_generator = buffer.step_generator(self.belief_num_mini_batch)

            for sample in data_generator:
                difference, belief_grad_norm = self.update_belief(sample, agent_id)

                train_info["belief_difference_loss"] += difference.item()
                train_info["belief_grad_norm"] += belief_grad_norm

        num_updates = self.belief_epoch * self.belief_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def train_critic(self, buffers, value_normalizer=None):
        train_info = {}

        train_info["value_loss"] = 0
        train_info["critic_grad_norm"] = 0

        for _ in range(self.critic_epoch):
            data_generators = []
            for agent_id in range(self.num_agents):
                if self.use_recurrent_policy:
                    data_generator = buffers[agent_id].chunk_generator(self.actor_num_mini_batch, self.data_chunk_length)
                else:
                    data_generator = buffers[agent_id].step_generator(self.actor_num_mini_batch)
                data_generators.append(data_generator)

            for batches in self.share_generator(data_generators):
                value_loss, critic_grad_norm = self.update_critic(batches, value_normalizer=value_normalizer)

                train_info["value_loss"] += value_loss.item()
                train_info["critic_grad_norm"] += critic_grad_norm

        num_updates = self.critic_epoch * self.critic_num_mini_batch

        for k, _ in train_info.items():
            train_info[k] /= num_updates

        return train_info

    def train_actor(self, buffer, agent_id):
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = buffer.chunk_generator(self.actor_num_mini_batch, self.data_chunk_length)
            else:
                data_generator = buffer.step_generator(self.actor_num_mini_batch)

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update_actor(sample, agent_id)

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def share_param_train_actor(self, buffers):
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(self.num_agents):
                if self.use_recurrent_policy:
                    data_generator = buffers[agent_id].chunk_generator(self.actor_num_mini_batch, self.data_chunk_length)
                else:
                    data_generator = buffers[agent_id].step_generator(self.actor_num_mini_batch)
                data_generators.append(data_generator)

            for batches in self.share_generator(data_generators):
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update_actor(batches, 0)

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
    
    def share_param_train_belief(self, buffers):
        train_info = {}
        train_info["belief_difference_loss"] = 0
        train_info["belief_grad_norm"] = 0

        for _ in range(self.belief_epoch):
            data_generators = []
            for agent_id in range(self.num_agents):
                if self.use_recurrent_policy:
                    data_generator = buffers[agent_id].chunk_generator(self.belief_num_mini_batch, self.data_chunk_length)
                else:
                    data_generator = buffers[agent_id].step_generator(self.belief_num_mini_batch)
                data_generators.append(data_generator)

            for batches in self.share_generator(data_generators):
                difference, belief_grad_norm = self.update_belief(batches, 0)

                train_info["belief_difference_loss"] += difference.item()
                train_info["belief_grad_norm"] += belief_grad_norm

        num_updates = self.belief_epoch * self.belief_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
    
    def share_generator(self, data_generators):
        """if actor and critic use the same buffer, when actors have heterogeneous input, there will be exceptions in train_critic()."""
        for _ in range(self.actor_num_mini_batch):
            batches = {}
            for generator in data_generators:
                sample = next(generator)
                for key in sample:
                    if key not in batches:
                        batches[key] = []
                    batches[key].append(sample[key])
            for key in batches:
                if batches[key][0] is None:
                    batches[key] = None
                else:
                    batches[key] = np.concatenate(batches[key], axis=0)
            yield batches

    def prep_training(self):
        for i, actor in enumerate(self.actors):
            actor.train()
            if self.env_belief:
                self.agents[i].belief.train()
        self.critic.train()

    def prep_rollout(self):
        for i, actor in enumerate(self.actors):
            actor.eval()
            if self.env_belief:
                self.agents[i].belief.eval()
        self.critic.eval()

    def save(self, path):
        if self.share_param:
            self.agents[0].save(path)
        else:
            for agent_id in range(self.num_agents):
                self.agents[agent_id].save(os.path.join(path, str(agent_id)))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))

    def restore(self, path):
        # actor的参数还原是Agent中实现的
        if self.share_param:
            self.agents[0].restore(path)
        else:
            for agent_id in range(self.num_agents):
                self.agents[agent_id].restore(os.path.join(path, str(agent_id)))
        # 0328: critic在这里实现
        # 0329: critic不用实现
        if self.args.get('load_critic', True):
            self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth")))