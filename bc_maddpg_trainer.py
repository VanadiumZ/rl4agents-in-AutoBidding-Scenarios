#!/usr/bin/env python3
"""
BC + MADDPG Training: 行为克隆预训练后使用多智能体DDPG训练
中心化训练 + 分散执行，适合竞争性拍卖环境
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from collections import defaultdict, deque
import random

# Add auction_sim to path
sys.path.append('.')

from auction_sim import config
from auction_sim.config import CurriculumStage, CURRICULUM_CONFIGS
from auction_sim.bc_trainer import BCTrainer
from auction_sim.bc_data_collector import BCDataCollector
from auction_sim.ma_environment import MultiAgentAuctionEnv
from auction_sim.agents import TruthfulAgent, ConservativeAgent, AggressiveAgent, MultiAgentLearningAgent

class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck过程噪声，用于连续动作探索"""
    
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        
    def reset(self):
        self.state = np.ones(self.size) * self.mu
        
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state.copy()

class Actor(nn.Module):
    """分散执行的Actor网络"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # 输出范围[0,1]，后续需要缩放
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 最后一层用小权重初始化
        with torch.no_grad():
            self.network[-2].weight.uniform_(-3e-3, 3e-3)
            self.network[-2].bias.uniform_(-3e-3, 3e-3)
    
    def forward(self, obs):
        out = self.network(obs)
        # 缩放到拍卖环境合理范围 [0.1, 2.0]
        return 0.1 + out * 1.9

class Critic(nn.Module):
    """中心化训练的Critic网络，观察所有智能体的状态和动作"""
    
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=128):
        super().__init__()
        
        # 状态编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(total_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(total_action_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        # Q值网络
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, global_obs, global_actions):
        obs_features = self.obs_encoder(global_obs)
        action_features = self.action_encoder(global_actions)
        combined = torch.cat([obs_features, action_features], dim=-1)
        return self.q_network(combined)

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class MADDPGAgent:
    """单个MADDPG智能体"""
    
    def __init__(self, agent_id, obs_dim, action_dim, total_obs_dim, total_action_dim,
                 lr_actor=1e-4, lr_critic=3e-4, gamma=0.95, tau=0.01):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # 网络
        self.actor = Actor(obs_dim, action_dim)
        self.actor_target = Actor(obs_dim, action_dim)
        self.critic = Critic(total_obs_dim, total_action_dim)
        self.critic_target = Critic(total_obs_dim, total_action_dim)
        
        # 复制参数到目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 噪声
        self.noise = OrnsteinUhlenbeckNoise(action_dim)
        
        # 损失记录
        self.actor_losses = []
        self.critic_losses = []
    
    def get_action(self, obs, add_noise=True, noise_scale=1.0):
        """获取动作"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = self.actor(obs_tensor).squeeze(0).numpy()
            
            if add_noise:
                noise = self.noise.sample() * noise_scale
                action += noise
                action = np.clip(action, 0.5, 1.5)  # 与IPPO/MAPPO保持一致的动作范围
            
            return action
    
    def load_bc_weights(self, bc_state_dict):
        """加载BC预训练权重"""
        try:
            # 尝试将BC权重映射到Actor网络
            actor_weights = {}
            for key, value in bc_state_dict.items():
                if 'actor' in key or 'backbone' in key:
                    # 简单的键名映射
                    new_key = key.replace('actor.', 'network.')
                    new_key = new_key.replace('backbone.', 'network.')
                    if new_key.startswith('network.'):
                        actor_weights[new_key] = value
            
            # 部分加载，忽略不匹配的层
            self.actor.load_state_dict(actor_weights, strict=False)
            self.actor_target.load_state_dict(self.actor.state_dict())
            return True
        except Exception as e:
            print(f"  ⚠️ {self.agent_id}: BC权重加载失败 ({e})")
            return False
    
    def soft_update(self):
        """软更新目标网络"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class MADDPGTrainer:
    """MADDPG训练器"""
    
    def __init__(self, n_agents, obs_dim, action_dim, 
                 lr_actor=1e-4, lr_critic=3e-4, gamma=0.95, tau=0.01,
                 buffer_capacity=100000, batch_size=64):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        
        # 全局维度
        self.total_obs_dim = obs_dim * n_agents
        self.total_action_dim = action_dim * n_agents
        
        # 创建智能体
        self.agents = {}
        for i in range(n_agents):
            agent_id = f"Learning_{i}"
            self.agents[agent_id] = MADDPGAgent(
                agent_id, obs_dim, action_dim, self.total_obs_dim, self.total_action_dim,
                lr_actor, lr_critic, gamma, tau
            )
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': {agent_id: [] for agent_id in self.agents.keys()},
            'actor_losses': {agent_id: [] for agent_id in self.agents.keys()},
            'critic_losses': {agent_id: [] for agent_id in self.agents.keys()}
        }
    
    def get_action(self, agent_id, obs, add_noise=True, noise_scale=1.0):
        """获取指定智能体的动作"""
        return self.agents[agent_id].get_action(obs, add_noise, noise_scale)
    
    def store_transition(self, states, actions, rewards, next_states, dones):
        """存储转换到经验回放缓冲区"""
        # 将多智能体状态和动作展平
        global_state = np.concatenate(list(states.values()))
        # 修复动作维度处理
        action_list = []
        for aid in sorted(actions.keys()):
            action = actions[aid]
            if isinstance(action, np.ndarray) and len(action) > 0:
                action_list.append(action[0])  # 取第一个元素
            else:
                action_list.append(float(action))
        global_action = np.array(action_list)
        global_reward = sum(rewards.values())
        global_next_state = np.concatenate(list(next_states.values()))
        global_done = any(dones.values())
        
        self.replay_buffer.push(global_state, global_action, global_reward, 
                               global_next_state, global_done)
    
    def update_agents(self):
        """更新所有智能体"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)  # [batch_size, n_agents]
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 更新每个智能体
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            # 当前智能体的观察（从全局状态中提取）
            agent_obs = states[:, i*self.obs_dim:(i+1)*self.obs_dim]
            agent_next_obs = next_states[:, i*self.obs_dim:(i+1)*self.obs_dim]
            
            # 计算下一个动作（使用目标actor）
            next_actions = torch.zeros_like(actions)
            for j, (aid, a) in enumerate(self.agents.items()):
                next_agent_obs = next_states[:, j*self.obs_dim:(j+1)*self.obs_dim]
                next_action = a.actor_target(next_agent_obs)
                if next_action.dim() > 1:
                    next_action = next_action.squeeze(-1)
                next_actions[:, j] = next_action
            
            # Critic损失
            with torch.no_grad():
                target_q = agent.critic_target(next_states, next_actions)
                target_q = rewards + (self.agents[agent_id].gamma * target_q * (1 - dones))
            
            current_q = agent.critic(states, actions)
            critic_loss = nn.MSELoss()(current_q, target_q)
            
            # 更新Critic
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
            agent.critic_optimizer.step()
            
            # Actor损失
            # 构造当前策略的动作
            policy_actions = torch.zeros_like(actions)
            for j, (aid, a) in enumerate(self.agents.items()):
                agent_obs_j = states[:, j*self.obs_dim:(j+1)*self.obs_dim]
                if j == i:  # 当前智能体用当前actor
                    current_action = agent.actor(agent_obs_j)
                    if current_action.dim() > 1:
                        current_action = current_action.squeeze(-1)
                    policy_actions[:, j] = current_action
                else:  # 其他智能体用实际动作
                    policy_actions[:, j] = actions[:, j]
            
            actor_loss = -agent.critic(states, policy_actions).mean()
            
            # 更新Actor
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_optimizer.step()
            
            # 软更新目标网络
            agent.soft_update()
            
            # 记录损失
            agent.actor_losses.append(actor_loss.item())
            agent.critic_losses.append(critic_loss.item())
    
    def train_episode(self, env, max_steps=1000):
        """训练一个episode"""
        states = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in self.agents.keys()}
        
        # 重置噪声
        for agent in self.agents.values():
            agent.noise.reset()
        
        for step in range(max_steps):
            # 获取动作
            actions = {}
            for agent_id in self.agents.keys():
                action = self.get_action(agent_id, states[agent_id], add_noise=True, 
                                       noise_scale=max(0.1, 1.0 - step/max_steps))  # 逐渐降低噪声
                # 环境期望数组格式 actions[agent_id][0]
                if isinstance(action, (float, np.float32, np.float64)):
                    actions[agent_id] = np.array([action])
                else:
                    actions[agent_id] = action
            
            # 执行动作
            step_result = env.step(actions)
            if len(step_result) == 5:
                next_states, rewards, terminated, truncated, infos = step_result
                dones = {k: terminated[k] or truncated[k] for k in terminated.keys()}
            else:
                next_states, rewards, dones, infos = step_result
            
            # 存储经验
            self.store_transition(states, actions, rewards, next_states, dones)
            
            # 更新统计
            for agent_id in self.agents.keys():
                episode_rewards[agent_id] += rewards[agent_id]
            
            states = next_states
            
            # 更新网络
            if step % 4 == 0:  # 每4步更新一次
                self.update_agents()
            
            if any(dones.values()):
                break
        
        # 记录episode统计
        for agent_id in self.agents.keys():
            self.training_stats['episode_rewards'][agent_id].append(episode_rewards[agent_id])
        
        # episode_wins不再使用，返回空dict保持接口一致
        return episode_rewards, {agent_id: 0 for agent_id in self.agents.keys()}
    
    def save_models(self, save_dir):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            torch.save(agent.actor.state_dict(), f"{save_dir}/{agent_id}_actor.pth")
            torch.save(agent.critic.state_dict(), f"{save_dir}/{agent_id}_critic.pth")
        
        print(f"MADDPG模型已保存到: {save_dir}")

class BCMADDPGTrainer:
    """BC预训练 + MADDPG训练的混合trainer"""
    
    def __init__(self,
                 use_bc_pretraining: bool = True,
                 bc_episodes: int = 10,
                 maddpg_episodes: int = 300,
                 target_stage: CurriculumStage = CurriculumStage.STAGE_8):
        
        self.use_bc_pretraining = use_bc_pretraining
        self.bc_episodes = bc_episodes
        self.maddpg_episodes = maddpg_episodes
        self.target_stage = target_stage
        
        # 路径配置
        self.bc_model_path = "auction_sim/models/bc_pretrained.pth"
        self.models_dir = "auction_sim/models/bc_maddpg"
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"BC + MADDPG Trainer初始化:")
        print(f"  BC预训练: {use_bc_pretraining}")
        print(f"  MADDPG训练: {maddpg_episodes} episodes")
        print(f"  目标环境: {target_stage.name}")
        print(f"  关键特性: 中心化训练 + 分散执行")
    
    def step1_bc_pretraining(self):
        """步骤1: BC预训练"""
        if not self.use_bc_pretraining:
            return None, None, None
        
        print("\n" + "="*70)
        print("步骤1: BC预训练")
        print("="*70)
        
        if os.path.exists(self.bc_model_path):
            print(f"发现已存在的BC模型: {self.bc_model_path}")
            print("跳过BC预训练，直接使用现有模型")
            
            try:
                bc_state_dict = torch.load(self.bc_model_path)
                print(f"✅ BC模型加载成功，包含 {len(bc_state_dict)} 个参数")
                return None, {'status': 'loaded_existing'}, {'model_path': self.bc_model_path}
            except Exception as e:
                print(f"❌ BC模型加载失败: {e}")
        
        return None, None, None
    
    def step2_maddpg_training(self):
        """步骤2: MADDPG训练"""
        print("\n" + "="*70)
        print("步骤2: MADDPG训练")
        print("="*70)
        
        # 环境配置
        stage_config = CURRICULUM_CONFIGS[self.target_stage]
        print(f"目标环境: {stage_config['description']}")
        
        # 创建智能体和环境
        rule_agents = self._create_rule_agents(stage_config)
        learning_agent_ids = [f"Learning_{i}" for i in range(stage_config['n_learning'])]
        
        env = MultiAgentAuctionEnv(
            learning_agent_ids,
            rule_agents,
            curriculum_stage=self.target_stage
        )
        
        # 创建MADDPG trainer
        maddpg = MADDPGTrainer(
            n_agents=stage_config['n_learning'],
            obs_dim=7,
            action_dim=1,
            lr_actor=1e-4,
            lr_critic=3e-4,
            gamma=0.95,
            tau=0.01,
            buffer_capacity=100000,
            batch_size=64
        )
        
        print(f"\n✨ MADDPG关键特性:")
        print("  - 中心化Critic：观察所有智能体状态和动作")
        print("  - 分散执行Actor：每个智能体独立决策")
        print("  - 连续动作空间：适合拍卖出价场景")
        print("  - OU噪声探索：比随机噪声更适合连续控制")
        
        # BC权重初始化
        if self.use_bc_pretraining and os.path.exists(self.bc_model_path):
            print("\n加载BC预训练权重...")
            bc_state_dict = torch.load(self.bc_model_path)
            
            for agent_id, agent in maddpg.agents.items():
                success = agent.load_bc_weights(bc_state_dict)
                if success:
                    print(f"  ✅ {agent_id}: BC权重加载成功")
        
        # 创建环境接口
        learning_agents = []
        for agent_id in learning_agent_ids:
            agent = MultiAgentLearningAgent(
                agent_id=agent_id,
                budget=stage_config['budget'],
                perception_noise_std=config.AGENT_PERCEPTION_NOISE_STD,
                model=None,
                is_training=True
            )
            
            # 包装MADDPG agent
            class MADDPGWrapper:
                def __init__(self, maddpg_trainer, agent_id):
                    self.maddpg_trainer = maddpg_trainer
                    self.agent_id = agent_id
                
                def predict(self, obs, deterministic=False):
                    action = self.maddpg_trainer.get_action(
                        self.agent_id, obs, 
                        add_noise=not deterministic
                    )
                    # 确保返回数组格式
                    if isinstance(action, (float, np.float32, np.float64)):
                        return np.array([action]), None
                    else:
                        return action, None
            
            agent.set_model(MADDPGWrapper(maddpg, agent_id))
            learning_agents.append(agent)
        
        # 训练循环
        print(f"\n开始MADDPG训练 {self.maddpg_episodes} episodes...")
        
        individual_performance = {
            agent_id: {
                'rewards': [],
                'win_rates': [],
                'rois': [],
                'budget_usage': []
            } for agent_id in learning_agent_ids
        }
        
        episode_metrics = []
        best_avg_reward = float('-inf')
        
        pbar = tqdm(range(self.maddpg_episodes), desc="MADDPG Training")
        
        for episode in pbar:
            # 训练一个episode
            ep_rewards, ep_wins = maddpg.train_episode(env, max_steps=stage_config['max_rounds'])
            
            # 计算统计
            total_reward = sum(ep_rewards.values())
            episode_stats = self._calculate_episode_stats(env, learning_agent_ids, stage_config, ep_rewards, ep_wins)
            episode_metrics.append(episode_stats)
            
            # 记录个体表现
            for agent_id in learning_agent_ids:
                if agent_id in episode_stats['individual_stats']:
                    stats = episode_stats['individual_stats'][agent_id]
                    individual_performance[agent_id]['rewards'].append(ep_rewards.get(agent_id, 0))
                    individual_performance[agent_id]['win_rates'].append(stats['win_rate'])
                    individual_performance[agent_id]['rois'].append(stats['roi'])
                    individual_performance[agent_id]['budget_usage'].append(stats['budget_usage'])
            
            # 更新进度条
            if len(episode_metrics) > 0:
                last_metrics = episode_metrics[-1]
                pbar.set_postfix({
                    'Reward': f"{total_reward:.0f}",
                    'WinRate': f"{last_metrics['avg_win_rate']:.1%}",
                    'L0_WR': f"{last_metrics['individual_stats'].get('Learning_0', {}).get('win_rate', 0):.1%}",
                    'L1_WR': f"{last_metrics['individual_stats'].get('Learning_1', {}).get('win_rate', 0):.1%}"
                })
            
            # 保存最佳模型
            if total_reward > best_avg_reward:
                best_avg_reward = total_reward
                maddpg.save_models(f"{self.models_dir}/best")
            
            # 定期保存
            if (episode + 1) % 50 == 0:
                maddpg.save_models(f"{self.models_dir}/checkpoint_ep{episode+1}")
                self._print_progress(episode + 1, episode_metrics[-20:], individual_performance)
        
        pbar.close()
        
        # 保存最终模型
        maddpg.save_models(f"{self.models_dir}/final")
        
        # 生成分析
        self._analyze_results(individual_performance)
        
        return maddpg, learning_agents, rule_agents, episode_metrics, individual_performance
    
    def _create_rule_agents(self, stage_config):
        """创建规则智能体"""
        rule_agents = []
        agent_id_counter = 0
        
        for _ in range(stage_config.get('n_truthful', 0)):
            agent = TruthfulAgent(
                f"Truthful_{agent_id_counter}",
                stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD
            )
            rule_agents.append(agent)
            agent_id_counter += 1
        
        for _ in range(stage_config.get('n_conservative', 0)):
            agent = ConservativeAgent(
                f"Conservative_{agent_id_counter}",
                stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD,
                stage_config['max_rounds']
            )
            rule_agents.append(agent)
            agent_id_counter += 1
        
        for _ in range(stage_config.get('n_aggressive', 0)):
            total_agents = sum([stage_config.get(f'n_{t}', 0) for t in ['learning', 'truthful', 'conservative', 'aggressive']])
            agent = AggressiveAgent(
                f"Aggressive_{agent_id_counter}",
                stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD,
                total_agents
            )
            rule_agents.append(agent)
            agent_id_counter += 1
        
        return rule_agents
    
    def _calculate_episode_stats(self, env, learning_agent_ids, stage_config, ep_rewards, ep_wins):
        """计算episode统计"""
        stats = {
            'avg_win_rate': 0.0,
            'avg_roi': 0.0,
            'budget_usage_ratio': 0.0,
            'individual_stats': {}
        }
        
        for agent_id in learning_agent_ids:
            agent_stats = {
                'win_rate': 0.0,
                'roi': -100.0,
                'budget_usage': 0.0
            }
            
            # 计算预算使用率
            initial_budget = stage_config['budget']
            remaining = env.agent_budgets.get(agent_id, initial_budget)
            used = initial_budget - remaining
            agent_stats['budget_usage'] = used / initial_budget
            
            # 计算胜率和ROI
            if hasattr(env, 'agent_histories') and agent_id in env.agent_histories:
                history = env.agent_histories[agent_id]
                if history:
                    total_profit = sum(h.get('profit', 0) for h in history)
                    total_cost = sum(h.get('cost', 0) for h in history)
                    wins = sum(1 for h in history if h.get('won', False))
                    
                    # 使用实际轮数而非history长度计算胜率
                    actual_rounds = max(h.get('round', 0) for h in history) + 1 if history else 0
                    agent_stats['win_rate'] = wins / actual_rounds if actual_rounds > 0 else 0.0
                    if total_cost > 0:
                        agent_stats['roi'] = (total_profit / total_cost) * 100
            
            stats['individual_stats'][agent_id] = agent_stats
            stats['avg_win_rate'] += agent_stats['win_rate']
            stats['avg_roi'] += agent_stats['roi']
            stats['budget_usage_ratio'] += agent_stats['budget_usage']
        
        # 平均化
        n_agents = len(learning_agent_ids)
        if n_agents > 0:
            stats['avg_win_rate'] /= n_agents
            stats['avg_roi'] /= n_agents
            stats['budget_usage_ratio'] /= n_agents
        
        return stats
    
    def _print_progress(self, episode, recent_metrics, individual_performance):
        """打印训练进度"""
        avg_win_rate = np.mean([m['avg_win_rate'] for m in recent_metrics])
        avg_roi = np.mean([m['avg_roi'] for m in recent_metrics])
        avg_budget = np.mean([m['budget_usage_ratio'] for m in recent_metrics])
        
        print(f"\nEpisode {episode}/{self.maddpg_episodes} (最近20集平均):")
        print(f"  平均胜率: {avg_win_rate:.1%}")
        print(f"  平均ROI: {avg_roi:.1f}%")
        print(f"  预算使用: {avg_budget:.1%}")
        
        # 个体表现
        print("\n  🔍 个体表现对比 (MADDPG中心化训练):")
        for agent_id in individual_performance.keys():
            recent_wrs = individual_performance[agent_id]['win_rates'][-20:]
            recent_rois = individual_performance[agent_id]['rois'][-20:]
            
            if recent_wrs:
                print(f"    {agent_id}: 胜率 {np.mean(recent_wrs):.1%}, ROI {np.mean(recent_rois):.0f}%")
        
        # 不对称度
        if len(individual_performance) == 2:
            agents = list(individual_performance.keys())
            wr_diff = abs(np.mean(individual_performance[agents[0]]['win_rates'][-20:]) - 
                         np.mean(individual_performance[agents[1]]['win_rates'][-20:]))
            print(f"\n  📊 不对称度: 胜率差异 {wr_diff:.1%}")
    
    def _analyze_results(self, individual_performance):
        """分析MADDPG结果"""
        print("\n" + "="*70)
        print("MADDPG结果分析")
        print("="*70)
        
        if len(individual_performance) == 2:
            agents = list(individual_performance.keys())
            
            # 最终性能
            final_episodes = 50
            agent0_final_wr = np.mean(individual_performance[agents[0]]['win_rates'][-final_episodes:])
            agent1_final_wr = np.mean(individual_performance[agents[1]]['win_rates'][-final_episodes:])
            
            agent0_final_roi = np.mean(individual_performance[agents[0]]['rois'][-final_episodes:])
            agent1_final_roi = np.mean(individual_performance[agents[1]]['rois'][-final_episodes:])
            
            print(f"\n最终性能对比 (最后{final_episodes}集):")
            print(f"  {agents[0]}: 胜率 {agent0_final_wr:.1%}, ROI {agent0_final_roi:.0f}%")
            print(f"  {agents[1]}: 胜率 {agent1_final_wr:.1%}, ROI {agent1_final_roi:.0f}%")
            
            wr_diff = abs(agent0_final_wr - agent1_final_wr)
            print(f"\n不对称度指标:")
            print(f"  胜率差异: {wr_diff:.1%}")
            
            # 算法对比
            print("\n算法对比:")
            print("  MAPPO: 胜率差异 ~35%")
            print("  IPPO:  胜率差异 20.1%")
            print(f"  MADDPG: 胜率差异 {wr_diff:.1%}")
            
            if wr_diff < 0.15:
                print("\n✅ MADDPG在解决不对称问题方面表现良好!")
            else:
                print("\n⚠️ MADDPG仍需进一步调优")
    
    def train_complete_pipeline(self):
        """执行完整训练流程"""
        print("="*80)
        print("BC + MADDPG TRAINING PIPELINE")
        print("中心化训练 + 分散执行")
        print("="*80)
        
        # 步骤1: BC预训练
        bc_trainer, bc_results, bc_metrics = self.step1_bc_pretraining()
        
        # 步骤2: MADDPG训练
        results = self.step2_maddpg_training()
        
        return {
            'bc_metrics': bc_metrics,
            'maddpg_results': results
        }

def main():
    """主函数"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("开始BC + MADDPG Training实验...")
    print("目标: 测试中心化训练能否更好地解决不对称问题")
    
    trainer = BCMADDPGTrainer(
        use_bc_pretraining=True,
        bc_episodes=10,
        maddpg_episodes=300,
        target_stage=CurriculumStage.STAGE_8
    )
    
    results = trainer.train_complete_pipeline()
    
    print("\n" + "="*80)
    print("BC + MADDPG 实验完成！")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()