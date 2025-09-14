# /auction_sim/ippo_trainer.py
"""
Independent PPO (IPPO) Trainer for k=2 scenario
Each learning agent trains independently without coordination
Better for competitive scenarios where agents should not cooperate
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Any
import os
from collections import deque
import matplotlib.pyplot as plt

class RunningMeanStd:
    """运行时均值和标准差计算，用于观测归一化"""
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
        
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
        
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

from . import config
from .ma_environment import MultiAgentAuctionEnv
from .agents import TruthfulAgent, ConservativeAgent, AggressiveAgent, MultiAgentLearningAgent

class IPPOActorCriticNetwork(nn.Module):
    """
    Independent Actor-Critic network for IPPO
    Each agent has completely separate network
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Independent backbone for each agent
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy) - output raw mean and log_std for Tanh-squash
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.full((action_dim,), -1.0))  # 初始化为-1.0降低探索方差
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
        # Action bounds for squashing
        self.action_low = 0.1
        self.action_high = 3.0
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Initialize actor mean layer with smaller weights
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        
    def forward(self, obs):
        features = self.backbone(obs)
        
        # Actor output - 直接输出[0.5, 1.5]范围的均值
        mean_raw = self.actor_mean(features)  # tanh -> [-1, 1]
        mean = 1.0 + 0.5 * torch.tanh(mean_raw)  # -> [0.5, 1.5]
        log_std = self.actor_logstd.clamp(-4, -1)  # 收紧std范围提升稳定性
        std = torch.exp(log_std)
        
        # Critic output
        value = self.critic(features)
        
        return mean, std, value
    
    def get_action(self, obs, deterministic=False):
        """统一采样接口：一次前向返回(action, log_prob, value)"""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        
        if deterministic:
            # 确定性动作：使用均值
            action = mean
        else:
            # 随机采样
            action = dist.rsample()  # 使用rsample支持重参数化
        
        # 两个分支都进行clamp，确保动作在有效范围内
        action = torch.clamp(action, self.action_low, self.action_high)
        
        # 计算log_prob（轻微失真可接受，配合较小std）
        if deterministic:
            log_prob = torch.zeros_like(action)
        else:
            log_prob = dist.log_prob(action)
            # 如果是多维动作，在最后一维求和
            if log_prob.dim() > 1:
                log_prob = log_prob.sum(-1)
        
        return action, log_prob, value
    
    def evaluate_action(self, obs, action):
        """评估动作的log_prob，与get_action保持一致"""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        
        # 先clamp再计算，与get_action保持一致
        action = torch.clamp(action, self.action_low, self.action_high)
        
        # 计算log_prob
        log_prob = dist.log_prob(action)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(-1)
        
        # 计算熵
        entropy = dist.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(-1)
        
        return log_prob, entropy, value

class IPPOTrainer:
    """
    Independent PPO Trainer - each agent learns completely independently
    """
    
    def __init__(self, 
                 obs_dim: int = 7,
                 action_dim: int = 1,
                 n_agents: int = 1,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.15,
                 vf_coef: float = 0.25,
                 ent_coef: float = 0.005,
                 max_grad_norm: float = 0.5,
                 max_buffer_size: int = 50000):
        
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.max_buffer_size = max_buffer_size
        
        # Create completely independent networks and optimizers for each agent
        self.networks = {}
        self.optimizers = {}
        # 添加观测归一化器
        self.obs_normalizers = {}
        
        for i in range(n_agents):
            agent_id = f"Learning_{i}"
            self.networks[agent_id] = IPPOActorCriticNetwork(obs_dim, action_dim)
            self.optimizers[agent_id] = optim.Adam(
                self.networks[agent_id].parameters(), lr=lr
            )
            self.obs_normalizers[agent_id] = RunningMeanStd(shape=(obs_dim,))
        
        # Independent experience buffers
        self.buffers = {agent_id: {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        } for agent_id in self.networks.keys()}
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': {agent_id: [] for agent_id in self.networks.keys()},
            'episode_lengths': [],
            'actor_losses': {agent_id: [] for agent_id in self.networks.keys()},
            'critic_losses': {agent_id: [] for agent_id in self.networks.keys()},
            'win_rates': {agent_id: [] for agent_id in self.networks.keys()}
        }
        
        print(f"IPPO Trainer initialized with {n_agents} independent agents")
    
    def get_action(self, agent_id: str, obs: np.ndarray, deterministic: bool = False):
        """Get action from independent policy network"""
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        if obs.ndim == 0:
            obs = obs.reshape(1,)
        
        # 归一化观测
        obs_normalized = self.obs_normalizers[agent_id].normalize(obs)
        obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = self.networks[agent_id].get_action(obs_tensor, deterministic)
            
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def store_experience(self, agent_id: str, obs, action, reward, value, log_prob, done):
        """Store experience in agent's independent buffer"""
        if len(self.buffers[agent_id]['observations']) >= self.max_buffer_size:
            for key in self.buffers[agent_id]:
                if isinstance(self.buffers[agent_id][key], list):
                    self.buffers[agent_id][key].pop(0)
        
        self.buffers[agent_id]['observations'].append(obs)
        self.buffers[agent_id]['actions'].append(action)
        self.buffers[agent_id]['rewards'].append(reward)
        self.buffers[agent_id]['values'].append(value)
        self.buffers[agent_id]['log_probs'].append(log_prob)
        self.buffers[agent_id]['dones'].append(done)
    
    def compute_gae(self, agent_id: str, next_value: float = 0.0):
        """Compute GAE for individual agent"""
        rewards = self.buffers[agent_id]['rewards']
        values = self.buffers[agent_id]['values'] + [next_value]
        dones = self.buffers[agent_id]['dones']
        
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def update_policy(self, agent_id: str, advantages, returns):
        """Update individual agent's policy using PPO"""
        # 更新观测归一化器并归一化观测
        obs_array = np.array(self.buffers[agent_id]['observations'])
        self.obs_normalizers[agent_id].update(obs_array)
        obs_normalized = np.array([self.obs_normalizers[agent_id].normalize(o) for o in obs_array])
        
        # Convert to tensors
        obs = torch.FloatTensor(obs_normalized)
        actions = torch.FloatTensor(self.buffers[agent_id]['actions'])
        old_log_probs = torch.FloatTensor(self.buffers[agent_id]['log_probs'])
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 0:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = advantages - adv_mean
        
        # 改进PPO更新：全量样本分批遍历，提升样本利用率
        n_epochs = 4
        batch_size = min(256, len(obs))
        n_batches = (len(obs) + batch_size - 1) // batch_size
        
        # 获取旧的价值函数输出用于裁剪
        with torch.no_grad():
            _, _, old_values = self.networks[agent_id].evaluate_action(obs, actions)
            old_values = old_values.squeeze(-1) if old_values.dim() > 1 else old_values
        
        for epoch in range(n_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(obs))
            
            # 分批遍历全量数据
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(obs))
                batch_indices = indices[start_idx:end_idx]
                
                obs_batch = obs[batch_indices]
                actions_batch = actions[batch_indices]
                old_log_probs_batch = old_log_probs[batch_indices]
                advantages_batch = advantages[batch_indices]
                returns_batch = returns[batch_indices]
                old_values_batch = old_values[batch_indices]
                
                # Forward pass
                log_probs, entropy, values = self.networks[agent_id].evaluate_action(
                    obs_batch, actions_batch
                )
                
                # PPO Actor loss
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 添加价值函数裁剪，防止critic过拟合
                if values.dim() > returns_batch.dim():
                    values = values.squeeze(-1)
                elif values.dim() < returns_batch.dim():
                    returns_batch = returns_batch.squeeze(-1)
                
                # Clipped value loss
                values_clipped = old_values_batch + torch.clamp(
                    values - old_values_batch, -self.clip_ratio, self.clip_ratio
                )
                vf_loss1 = (values - returns_batch).pow(2)
                vf_loss2 = (values_clipped - returns_batch).pow(2)
                critic_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
                
                # Optimize individual agent's network
                self.optimizers[agent_id].zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.networks[agent_id].parameters(), self.max_grad_norm)
                self.optimizers[agent_id].step()
                
                # Store losses
                self.training_stats['actor_losses'][agent_id].append(actor_loss.item())
                self.training_stats['critic_losses'][agent_id].append(critic_loss.item())
    
    def clear_buffers(self):
        """Clear experience buffers"""
        for agent_id in self.buffers:
            for key in self.buffers[agent_id]:
                self.buffers[agent_id][key] = []
    
    def train_episode(self, env: MultiAgentAuctionEnv, max_steps: int = 16000):
        """Train for one episode with independent learning"""
        obs = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in env.learning_agent_ids}
        episode_wins = {agent_id: 0 for agent_id in env.learning_agent_ids}
        
        for step in range(max_steps):
            actions = {}
            values = {}
            log_probs = {}
            
            # Get actions for all learning agents independently
            for agent_id in env.learning_agent_ids:
                action, log_prob, value = self.get_action(agent_id, obs[agent_id])
                
                actions[agent_id] = np.array([action])
                values[agent_id] = value
                log_probs[agent_id] = log_prob
            
            # Environment step
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Store experiences independently
            for agent_id in env.learning_agent_ids:
                done = terminated[agent_id] or truncated[agent_id]
                
                self.store_experience(
                    agent_id, obs[agent_id], actions[agent_id][0], 
                    rewards[agent_id], values[agent_id], log_probs[agent_id], done
                )
                
                episode_rewards[agent_id] += rewards[agent_id]
                
                # Track wins
                if hasattr(env, 'agent_histories') and agent_id in env.agent_histories:
                    if env.agent_histories[agent_id] and env.agent_histories[agent_id][-1].get('won', False):
                        episode_wins[agent_id] += 1
            
            obs = next_obs
            
            # Check if episode is done
            if any(terminated.values()) or any(truncated.values()):
                break
        
        # Update each agent's policy independently
        for agent_id in env.learning_agent_ids:
            # Get final value for GAE computation
            obs_normalized = self.obs_normalizers[agent_id].normalize(obs[agent_id])
            obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0)
            with torch.no_grad():
                _, _, next_value = self.networks[agent_id].get_action(obs_tensor, deterministic=True)
                next_value = next_value.item()
            
            advantages, returns = self.compute_gae(agent_id, next_value)
            self.update_policy(agent_id, advantages, returns)
            
            # 修正统计口径：计算正确的胜率和avg_bid_ratio
            actual_auctions = 0
            actual_wins = 0
            bid_ratios = []
            
            if hasattr(env, 'agent_histories') and agent_id in env.agent_histories:
                history = env.agent_histories[agent_id]
                actual_auctions = len(history)
                actual_wins = sum(1 for h in history if h.get('won', False))
                
                # 计算出价比率
                for h in history:
                    bid = h.get('bid', 0)
                    perceived_value = h.get('perceived_value', 0)
                    if perceived_value > 0:
                        bid_ratios.append(bid / perceived_value)
            
            # 修正胜率计算：使用实际拍卖次数而非max_steps
            win_rate = actual_wins / actual_auctions if actual_auctions > 0 else 0.0
            avg_bid_ratio = np.mean(bid_ratios) if bid_ratios else 1.0
            
            # Store episode statistics
            self.training_stats['episode_rewards'][agent_id].append(episode_rewards[agent_id])
            self.training_stats['win_rates'][agent_id].append(win_rate)
            
            # 添加avg_bid_ratio统计
            if 'avg_bid_ratios' not in self.training_stats:
                self.training_stats['avg_bid_ratios'] = {aid: [] for aid in env.learning_agent_ids}
            self.training_stats['avg_bid_ratios'][agent_id].append(avg_bid_ratio)
        
        self.training_stats['episode_lengths'].append(step + 1)
        self.clear_buffers()
        
        return episode_rewards, episode_wins
    
    def save_models(self, save_dir: str):
        """Save trained models and observation normalizers"""
        os.makedirs(save_dir, exist_ok=True)
        
        for agent_id, network in self.networks.items():
            torch.save(network.state_dict(), f"{save_dir}/{agent_id}_ippo_model.pth")
            
            # 保存观测归一化器状态
            normalizer_state = {
                'mean': self.obs_normalizers[agent_id].mean,
                'var': self.obs_normalizers[agent_id].var,
                'count': self.obs_normalizers[agent_id].count
            }
            np.save(f"{save_dir}/{agent_id}_obs_normalizer.npy", normalizer_state)
        
        print(f"IPPO models and normalizers saved to {save_dir}")
    
    def load_models(self, save_dir: str):
        """Load trained models and observation normalizers"""
        for agent_id, network in self.networks.items():
            model_path = f"{save_dir}/{agent_id}_ippo_model.pth"
            if os.path.exists(model_path):
                network.load_state_dict(torch.load(model_path))
                print(f"Loaded IPPO model for {agent_id}")
            
            # 加载观测归一化器状态
            normalizer_path = f"{save_dir}/{agent_id}_obs_normalizer.npy"
            if os.path.exists(normalizer_path):
                normalizer_state = np.load(normalizer_path, allow_pickle=True).item()
                self.obs_normalizers[agent_id].mean = normalizer_state['mean']
                self.obs_normalizers[agent_id].var = normalizer_state['var']
                self.obs_normalizers[agent_id].count = normalizer_state['count']
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        for agent_id in self.networks.keys():
            axes[0, 0].plot(self.training_stats['episode_rewards'][agent_id], 
                           label=f"{agent_id} (IPPO)", alpha=0.7)
        axes[0, 0].set_title('Episode Rewards (IPPO)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win rates
        for agent_id in self.networks.keys():
            axes[0, 1].plot(self.training_stats['win_rates'][agent_id], 
                           label=f"{agent_id} (IPPO)", alpha=0.7)
        axes[0, 1].set_title('Win Rates (IPPO)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Actor losses
        for agent_id in self.networks.keys():
            if self.training_stats['actor_losses'][agent_id]:
                axes[1, 0].plot(self.training_stats['actor_losses'][agent_id], 
                               label=f"{agent_id} (IPPO)", alpha=0.7)
        axes[1, 0].set_title('Actor Losses (IPPO)')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Critic losses
        for agent_id in self.networks.keys():
            if self.training_stats['critic_losses'][agent_id]:
                axes[1, 1].plot(self.training_stats['critic_losses'][agent_id], 
                               label=f"{agent_id} (IPPO)", alpha=0.7)
        axes[1, 1].set_title('Critic Losses (IPPO)')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"IPPO training curves saved to {save_path}")
        
        plt.show()