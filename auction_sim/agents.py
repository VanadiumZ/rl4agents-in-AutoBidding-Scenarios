# /auction_sim/agents.py
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import os
from . import config
from .training_monitor import TrainingMonitor


class ReplayBuffer:
    """经验回放缓冲区，用于存储和采样强化学习的经验数据"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加一条经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """随机采样一批经验"""
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self) -> int:
        return len(self.buffer)

class Agent(ABC):
    """所有智能体的抽象基类"""
    def __init__(self, agent_id: str, budget: float, perception_noise_std: float):
        self.id = agent_id
        self.initial_budget = budget
        self.budget = budget
        self.perception_noise_std = perception_noise_std
        
        # 用于记录历史数据
        self.history = []

    def perceive(self, true_value: float) -> float:
        """模拟对真实价值的感知，加入噪声"""
        perceived_value = true_value + np.random.normal(0, self.perception_noise_std)
        return max(0, perceived_value) # 感知价值不能为负

    @abstractmethod
    def bid(self, perceived_value: float) -> float:
        """
        核心出价方法，由子类实现。
        """
        pass

    def can_afford_bid(self, bid_price: float) -> bool:
        """检查智能体是否有足够预算支付出价"""
        return self.budget >= bid_price

    def update(self, result: Dict, round_num: int, true_value: float = None, profit: float = None):
        """
        根据一轮拍卖的结果更新自身状态。
        """
        if result and result['won']:
            # Cost_t应该是 cost-per-click * CTR
            expected_cost = result['cost_per_click'] * result['slot_ctr']
            cost = min(expected_cost, self.budget)
            self.budget -= cost
        else:
            cost = 0.0
        
        # 记录本轮数据，包括利润
        self.history.append({
            'round': round_num,
            'result': result,
            'cost': cost,
            'budget': self.budget,
            'true_value': true_value,
            'profit': profit if profit is not None else 0.0,
        })

    def get_cumulative_profit(self) -> float:
        """计算累计利润"""
        return sum(record['profit'] for record in self.history)
    
    def get_total_cost(self) -> float:
        """计算总花费"""
        return sum(record['cost'] for record in self.history)
    
    def get_roi(self) -> float:
        """计算ROI = 累计利润 / 累计成本 * 100%"""
        total_cost = self.get_total_cost()
        if total_cost == 0:
            return 0.0
        return (self.get_cumulative_profit() / total_cost) * 100

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, budget={self.budget:.2f})"

# --- 规则智能体 ---

class TruthfulAgent(Agent):
    """“老实人”智能体：出价等于感知价值"""
    def bid(self, perceived_value: float) -> float:
        return perceived_value

class ConservativeAgent(Agent):
    """“保守派”智能体：根据预算消耗节奏调整出价"""
    def __init__(self, agent_id: str, budget: float, perception_noise_std: float, total_rounds: int):
        super().__init__(agent_id, budget, perception_noise_std)
        self.total_rounds = total_rounds
        self.alpha = 1.0  # 初始调整因子

    def bid(self, perceived_value: float) -> float:
        current_round = len(self.history) + 1
        if current_round <= 1:
            return perceived_value * self.alpha

        # 1. 计算理想花费速度
        ideal_pace = self.initial_budget / self.total_rounds
        
        # 2. 计算实际花费速度
        budget_spent = self.initial_budget - self.budget
        actual_pace = budget_spent / (current_round -1)

        # 3. 计算并平滑更新alpha
        if actual_pace < 1e-6: # 避免除以0
            new_alpha = 1.1 # 如果还没花钱，就稍微激进一点
        else:
            # 使用一个平滑函数来更新alpha，防止剧烈波动
            target_alpha = ideal_pace / actual_pace
            new_alpha = config.CONSERVATIVE_AGENT_SMOOTHING * self.alpha + \
                        (1 - config.CONSERVATIVE_AGENT_SMOOTHING) * target_alpha
        
        self.alpha = max(0.1, min(new_alpha, 3.0)) # 限制alpha范围

        return perceived_value * self.alpha

class AggressiveAgent(Agent):
    """“激进派”智能体：根据近期胜率调整出价"""
    def __init__(self, agent_id: str, budget: float, perception_noise_std: float, n_agents: int):
        super().__init__(agent_id, budget, perception_noise_std)
        self.target_win_rate = 1.0 / n_agents
        self.win_history = deque(maxlen=config.AGGRESSIVE_AGENT_LOOKBACK)
        self.beta = 1.0

    def bid(self, perceived_value: float) -> float:
        if not self.win_history:
            current_win_rate = 0.0
        else:
            current_win_rate = sum(self.win_history) / len(self.win_history)
        
        # 更新好胜因子 beta
        self.beta = 1.0 + config.AGGRESSIVE_AGENT_LAMBDA * (self.target_win_rate - current_win_rate)
        self.beta = max(0.5, min(self.beta, 2.5)) # 限制beta范围

        return perceived_value * self.beta

    def update(self, result: Dict, round_num: int, true_value: float = None, profit: float = None):
        super().update(result, round_num, true_value, profit)
        self.win_history.append(1 if result and result['won'] else 0)

# --- 学习智能体 (框架) ---

class LearningAgent(Agent):
    """"自适应"学习智能体：使用强化学习模型进行出价"""
    # PPO算法
    def __init__(self, agent_id: str, budget: float, perception_noise_std: float, monitor: Optional[TrainingMonitor] = None):
        super().__init__(agent_id, budget, perception_noise_std)
        
        # 设置模型保存路径
        self.model_dir = "models/ppo"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 训练监控器
        self.monitor = monitor
        if self.monitor is None:
            self.monitor = TrainingMonitor(output_dir="training_logs/ppo")
            
        # 累计奖励
        self.total_reward = 0.0
        self.round_num = 0
        
        # 导入PPO相关库
        from stable_baselines3 import PPO
        from stable_baselines3.common.policies import ActorCriticPolicy
        import torch as th
        import torch.nn as nn
        import gym
        from gym import spaces
        
        # 创建自定义环境
        class BiddingEnv(gym.Env):
            def __init__(self):
                super().__init__()
                # 定义动作空间：连续动作空间，输出范围为[-1, 1]，由tanh激活函数保证
                self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                
                # 定义状态空间：5个连续状态变量
                self.observation_space = spaces.Box(
                    low=np.array([0, 0, 0, 0, 0]),  # 最小值
                    high=np.array([20, 1, 1, 1, 20]),  # 最大值
                    shape=(5,),
                    dtype=np.float32
                )
                
                # 初始化状态
                self.state = np.zeros(5, dtype=np.float32)
                self.reset()
            
            def reset(self):
                # 重置环境状态
                self.state = np.array([
                    0.0,  # perceived_value_t
                    1.0,  # remaining_budget_ratio
                    1.0,  # time_ratio
                    0.5,  # recent_win_rate
                    0.0,  # recent_avg_profit_per_win
                ], dtype=np.float32)
                return self.state
            
            def step(self, action):
                # 在实际使用中，这个方法不会被直接调用
                # 我们会在agent.update()中手动处理状态转换和奖励计算
                return self.state, 0.0, False, {}
        
        # 创建环境实例
        self.env = BiddingEnv()
        
        # 创建PPO模型
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=0.0003,
            n_steps=64,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0
        )
        
        # 存储当前状态、动作和奖励
        self.current_state = None
        self.current_action = None
        self.current_perceived_value = None
        
        # 用于计算状态的辅助变量
        self.win_history = deque(maxlen=20)  # 存储最近20轮的获胜情况
        self.profit_history = deque(maxlen=20)  # 存储最近20轮的利润
        self.total_rounds = config.SIMULATION_ROUNDS
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=1000)  # 存储1000条经验
        
        # 奖励权重
        self.w_profit = 1.0  # 利润奖励权重
        self.w_pacing = 0.2  # 预算节奏奖励权重
        self.w_efficiency = 0.3  # 成本效率奖励权重
        
        print(f"LearningAgent {self.id} initialized with PPO model.")
    
    def get_state(self, perceived_value: float) -> np.ndarray:
        """
        构建并返回当前的状态向量,用于输入RL模型。
        """
        current_round = len(self.history) + 1
        
        # 1. 当前感知价值
        normalized_perceived_value = perceived_value / 20.0  # 归一化到[0,1]范围
        
        # 2. 剩余预算比例
        remaining_budget_ratio = self.budget / self.initial_budget
        
        # 3. 剩余时间比例
        time_ratio = (self.total_rounds - current_round) / self.total_rounds
        
        # 4. 最近的胜率
        if not self.win_history:
            recent_win_rate = 0.5  # 初始默认值
        else:
            recent_win_rate = sum(self.win_history) / len(self.win_history)
        
        # 5. 最近平均每次获胜的利润
        if not self.profit_history or sum(self.win_history) == 0:
            recent_avg_profit_per_win = 0.0
        else:
            # 只计算获胜轮次的平均利润
            win_profits = [p for p, w in zip(self.profit_history, self.win_history) if w]
            if win_profits:
                recent_avg_profit_per_win = sum(win_profits) / len(win_profits)
            else:
                recent_avg_profit_per_win = 0.0
        
        # 构建状态向量
        state = np.array([
            normalized_perceived_value,  # perceived_value_t
            remaining_budget_ratio,      # remaining_budget_ratio
            time_ratio,                  # time_ratio
            recent_win_rate,             # recent_win_rate
            recent_avg_profit_per_win,   # recent_avg_profit_per_win
        ], dtype=np.float32)
        
        return state

    def bid(self, perceived_value: float) -> float:
        """
        使用RL模型预测动作并转化为出价。
        """

        # PPO算法
        # 1. 获取当前状态
        self.current_perceived_value = perceived_value
        self.current_state = self.get_state(perceived_value)
        
        try:
            # 2. 使用模型预测动作
            action, _ = self.model.predict(self.current_state, deterministic=False)  # 训练时使用随机策略
            self.current_action = action
            
            # 3. 将动作缩放到合理的出价范围
            bid_price = self.scale_action(action[0], perceived_value)
        except Exception as e:
            # 如果模型预测失败，使用简化版本的出价策略
            print(f"Warning: Failed to predict with PPO model: {e}")
            print(f"Using simplified bidding strategy instead.")
            
            # 简化版本：基于状态的启发式规则
            state = self.current_state
            
            # 根据剩余预算和时间调整出价策略
            if state[1] < state[2]:  # 预算消耗过快
                bid_multiplier = 0.8  # 降低出价
            elif state[1] > state[2] * 1.2:  # 预算消耗过慢
                bid_multiplier = 1.2  # 提高出价
            else:  # 预算消耗适中
                bid_multiplier = 1.0  # 保持当前出价
            
            # 根据胜率调整
            if state[3] < 0.3:  # 胜率过低
                bid_multiplier += 0.2  # 提高出价
            elif state[3] > 0.7:  # 胜率过高
                bid_multiplier -= 0.2  # 降低出价
            
            # 添加一些随机性
            bid_multiplier += np.random.uniform(-0.1, 0.1)
            
            # 限制在合理范围内
            bid_multiplier = max(0.5, min(2.0, bid_multiplier))
            
            # 计算最终出价
            bid_price = perceived_value * bid_multiplier
            
            # 记录当前动作
            self.current_action = np.array([(bid_multiplier - 0.5) * 2 / 1.5 - 1])  # 映射回[-1,1]范围
        
        return bid_price

    def scale_action(self, raw_action: float, perceived_value: float) -> float:
        """将模型输出 (-1, 1) 映射到出价范围。"""
        # 将tanh输出映射到 [0.5, 2.0] * perceived_value
        # 这个范围可以根据实际情况调整
        min_multiplier = 0.5
        max_multiplier = 2.0
        
        # 线性映射: [-1, 1] -> [min_multiplier, max_multiplier]
        bid_multiplier = min_multiplier + (raw_action + 1) * (max_multiplier - min_multiplier) / 2
        
        # 计算最终出价
        bid_price = perceived_value * bid_multiplier
        
        return bid_price
    
    def update(self, result: Dict, round_num: int, true_value: float = None, profit: float = None):
        """
        更新智能体状态，并将 (s, a, r, s') 存入经验池。
        """
        # 更新智能体状态，计算奖励，并训练PPO模型

        # 首先调用父类的update方法更新基本状态
        super().update(result, round_num, true_value, profit)
        
        # 更新轮次计数
        self.round_num = round_num
        
        # 如果没有当前状态或动作（例如第一轮），则不进行学习
        if self.current_state is None or self.current_action is None:
            return
        
        # 更新历史记录
        won = result and result['won']
        self.win_history.append(1 if won else 0)
        self.profit_history.append(profit if profit is not None else 0.0)
        
        # 计算奖励
        reward = self.calculate_reward(result, profit)
        self.total_reward += reward
        
        # 获取下一个状态（这里简化处理，实际应该在下一轮bid时获取）
        next_state = self.get_state(self.current_perceived_value)
        
        # 将经验存入回放缓冲区
        self.replay_buffer.add(self.current_state, self.current_action, reward, next_state, False)
        
        # 计算当前的胜率和出价/价值比
        recent_win_rate = sum(self.win_history) / len(self.win_history) if self.win_history else 0
        bid_value_ratio = result['bid'] / true_value if result and true_value > 0 else 0
        budget_remaining_ratio = self.budget / self.initial_budget
        
        # 记录训练指标
        self.monitor.log_metrics(round_num, {
            'reward': reward,
            'profit': profit if profit is not None else 0.0,
            'win_rate': recent_win_rate,
            'bid_value_ratio': bid_value_ratio,
            'budget_remaining_ratio': budget_remaining_ratio
        })
        
        # 手动训练PPO模型（每10轮训练一次）
        if round_num % 10 == 0 and len(self.replay_buffer) >= 100:  # 积累一些经验后再开始训练
            try:
                # 使用经验回放进行训练
                self.model.learn(total_timesteps=10)
                
                # 每100轮保存模型
                if round_num % 100 == 0:
                    model_path = os.path.join(self.model_dir, f"ppo_model_round_{round_num}.zip")
                    self.model.save(model_path)
                    print(f"Model saved to {model_path}")
                    
                    # 绘制训练曲线
                    self.monitor.plot_training_curves(
                        output_file=os.path.join(self.monitor.output_dir, f"training_curves_round_{round_num}.png")
                    )
            except Exception as e:
                print(f"Warning: Failed to train PPO model: {e}")
                print(f"Continuing without training for this round.")
        
        # 重置当前状态和动作
        self.current_state = None
        self.current_action = None
        self.current_perceived_value = None
    
    def calculate_reward(self, result: Dict, profit: float) -> float:
        """
        计算复合奖励函数。
        """
        # 1. 直接利润奖励
        profit_reward = profit if profit is not None else 0.0
        
        # 2. 预算节奏奖励
        current_round = len(self.history)
        ideal_spend_per_round = self.initial_budget / self.total_rounds
        
        # 获取当前轮次的花费
        current_cost = self.history[-1]['cost'] if self.history else 0.0
        
        # 计算超支情况
        overspend = current_cost - ideal_spend_per_round
        pacing_reward = -max(0, overspend)
        
        # 3. 成本效率奖励
        if current_cost > 1e-8 and profit is not None and profit > 0:  # 避免除以零
            efficiency_reward = profit / (current_cost + 1e-8)
        else:
            efficiency_reward = 0.0
        
        # 组合奖励
        total_reward = (
            self.w_profit * profit_reward + 
            self.w_pacing * pacing_reward + 
            self.w_efficiency * efficiency_reward
        )
        
        return total_reward


class DDPGLearningAgent(Agent):
    """"自适应"学习智能体：使用DDPG强化学习模型进行出价"""
    def __init__(self, agent_id: str, budget: float, perception_noise_std: float, monitor: Optional[TrainingMonitor] = None):
        super().__init__(agent_id, budget, perception_noise_std)
        
        # 设置模型保存路径
        self.model_dir = "models/ddpg"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 训练监控器
        self.monitor = monitor
        if self.monitor is None:
            self.monitor = TrainingMonitor(output_dir="training_logs/ddpg")
            
        # 累计奖励
        self.total_reward = 0.0
        self.round_num = 0
        
        # 导入DDPG相关库
        from stable_baselines3 import DDPG
        from stable_baselines3.common.noise import NormalActionNoise
        import torch as th
        import torch.nn as nn
        import gym
        from gym import spaces
        
        # 创建自定义环境
        class BiddingEnv(gym.Env):
            def __init__(self):
                super().__init__()
                # 定义动作空间：连续动作空间，输出范围为[-1, 1]
                self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                
                # 定义状态空间：5个连续状态变量
                self.observation_space = spaces.Box(
                    low=np.array([0, 0, 0, 0, 0]),  # 最小值
                    high=np.array([20, 1, 1, 1, 20]),  # 最大值
                    shape=(5,),
                    dtype=np.float32
                )
                
                # 初始化状态
                self.state = np.zeros(5, dtype=np.float32)
                self.reset()
            
            def reset(self):
                # 重置环境状态
                self.state = np.array([
                    0.0,  # perceived_value_t
                    1.0,  # remaining_budget_ratio
                    1.0,  # time_ratio
                    0.5,  # recent_win_rate
                    0.0,  # recent_avg_profit_per_win
                ], dtype=np.float32)
                return self.state
            
            def step(self, action):
                # 在实际使用中，这个方法不会被直接调用
                # 我们会在agent.update()中手动处理状态转换和奖励计算
                return self.state, 0.0, False, {}
        
        # 创建环境实例
        self.env = BiddingEnv()
        
        # 为DDPG添加动作噪声，促进探索
        n_actions = self.env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        
        # 创建DDPG模型
        self.model = DDPG(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=0.001,
            buffer_size=10000,  # 经验回放缓冲区大小
            learning_starts=100,  # 开始学习前收集的样本数
            batch_size=64,
            tau=0.005,  # 目标网络软更新参数
            gamma=0.99,  # 折扣因子
            action_noise=action_noise,
            verbose=0
        )
        
        # 存储当前状态、动作和奖励
        self.current_state = None
        self.current_action = None
        self.current_perceived_value = None
        
        # 用于计算状态的辅助变量
        self.win_history = deque(maxlen=20)  # 存储最近20轮的获胜情况
        self.profit_history = deque(maxlen=20)  # 存储最近20轮的利润
        self.total_rounds = config.SIMULATION_ROUNDS
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=10000)  # 存储(s, a, r, s', done)元组
        
        # 奖励权重
        self.w_profit = 1.0  # 利润奖励权重
        self.w_pacing = 0.2  # 预算节奏奖励权重
        self.w_efficiency = 0.3  # 成本效率奖励权重
        
        print(f"DDPGLearningAgent {self.id} initialized with DDPG model.")
    
    def get_state(self, perceived_value: float) -> np.ndarray:
        """
        构建并返回当前的状态向量,用于输入RL模型。
        """
        current_round = len(self.history) + 1
        
        # 1. 当前感知价值
        normalized_perceived_value = perceived_value / 20.0  # 归一化到[0,1]范围
        
        # 2. 剩余预算比例
        remaining_budget_ratio = self.budget / self.initial_budget
        
        # 3. 剩余时间比例
        time_ratio = (self.total_rounds - current_round) / self.total_rounds
        
        # 4. 最近的胜率
        if not self.win_history:
            recent_win_rate = 0.5  # 初始默认值
        else:
            recent_win_rate = sum(self.win_history) / len(self.win_history)
        
        # 5. 最近平均每次获胜的利润
        if not self.profit_history or sum(self.win_history) == 0:
            recent_avg_profit_per_win = 0.0
        else:
            # 只计算获胜轮次的平均利润
            win_profits = [p for p, w in zip(self.profit_history, self.win_history) if w]
            if win_profits:
                recent_avg_profit_per_win = sum(win_profits) / len(win_profits)
            else:
                recent_avg_profit_per_win = 0.0
        
        # 构建状态向量
        state = np.array([
            normalized_perceived_value,  # perceived_value_t
            remaining_budget_ratio,      # remaining_budget_ratio
            time_ratio,                  # time_ratio
            recent_win_rate,             # recent_win_rate
            recent_avg_profit_per_win,   # recent_avg_profit_per_win
        ], dtype=np.float32)
        
        return state

    def bid(self, perceived_value: float) -> float:
        """
        使用DDPG模型预测动作并转化为出价。
        """
        # 1. 获取当前状态
        self.current_perceived_value = perceived_value
        self.current_state = self.get_state(perceived_value)
        
        try:
            # 2. 使用模型预测动作
            action, _ = self.model.predict(self.current_state, deterministic=False)  # 训练时使用随机策略
            self.current_action = action
            
            # 3. 将动作缩放到合理的出价范围
            bid_price = self.scale_action(action[0], perceived_value)
        except Exception as e:
            # 如果模型预测失败，使用简化版本的出价策略
            print(f"Warning: Failed to predict with DDPG model: {e}")
            print(f"Using simplified bidding strategy instead.")
            
            # 简化版本：基于状态的启发式规则
            state = self.current_state
            
            # 根据剩余预算和时间调整出价策略
            if state[1] < state[2]:  # 预算消耗过快
                bid_multiplier = 0.8  # 降低出价
            elif state[1] > state[2] * 1.2:  # 预算消耗过慢
                bid_multiplier = 1.2  # 提高出价
            else:  # 预算消耗适中
                bid_multiplier = 1.0  # 保持当前出价
            
            # 根据胜率调整
            if state[3] < 0.3:  # 胜率过低
                bid_multiplier += 0.2  # 提高出价
            elif state[3] > 0.7:  # 胜率过高
                bid_multiplier -= 0.2  # 降低出价
            
            # 添加一些随机性
            bid_multiplier += np.random.uniform(-0.1, 0.1)
            
            # 限制在合理范围内
            bid_multiplier = max(0.5, min(2.0, bid_multiplier))
            
            # 计算最终出价
            bid_price = perceived_value * bid_multiplier
            
            # 记录当前动作
            self.current_action = np.array([(bid_multiplier - 0.5) * 2 / 1.5 - 1])  # 映射回[-1,1]范围
        
        return bid_price

    def scale_action(self, raw_action: float, perceived_value: float) -> float:
        """将模型输出 (-1, 1) 映射到出价范围。"""
        # 将tanh输出映射到 [0.5, 2.0] * perceived_value
        # 这个范围可以根据实际情况调整
        min_multiplier = 0.5
        max_multiplier = 2.0
        
        # 线性映射: [-1, 1] -> [min_multiplier, max_multiplier]
        bid_multiplier = min_multiplier + (raw_action + 1) * (max_multiplier - min_multiplier) / 2
        
        # 计算最终出价
        bid_price = perceived_value * bid_multiplier
        
        return bid_price
    
    def update(self, result: Dict, round_num: int, true_value: float = None, profit: float = None):
        """
        更新智能体状态，并将 (s, a, r, s') 存入经验池。
        """
        # 首先调用父类的update方法更新基本状态
        super().update(result, round_num, true_value, profit)
        
        # 更新轮次计数
        self.round_num = round_num
        
        # 如果没有当前状态或动作（例如第一轮），则不进行学习
        if self.current_state is None or self.current_action is None:
            return
        
        # 更新历史记录
        won = result and result['won']
        self.win_history.append(1 if won else 0)
        self.profit_history.append(profit if profit is not None else 0.0)
        
        # 计算奖励
        reward = self.calculate_reward(result, profit)
        self.total_reward += reward
        
        # 获取下一个状态（这里简化处理，实际应该在下一轮bid时获取）
        next_state = self.get_state(self.current_perceived_value)
        
        # 将经验存入回放缓冲区
        self.replay_buffer.add(self.current_state, self.current_action, reward, next_state, False)
        
        # 计算当前的胜率和出价/价值比
        recent_win_rate = sum(self.win_history) / len(self.win_history) if self.win_history else 0
        bid_value_ratio = result['bid'] / true_value if result and true_value > 0 else 0
        budget_remaining_ratio = self.budget / self.initial_budget
        
        # 记录训练指标
        self.monitor.log_metrics(round_num, {
            'reward': reward,
            'profit': profit if profit is not None else 0.0,
            'win_rate': recent_win_rate,
            'bid_value_ratio': bid_value_ratio,
            'budget_remaining_ratio': budget_remaining_ratio
        })
        
        # 手动训练DDPG模型（每10轮训练一次）
        if round_num % 10 == 0 and len(self.replay_buffer) >= 100:  # 积累一些经验后再开始训练
            try:
                # 从经验回放缓冲区采样批量数据
                batch_size = 64
                experiences = self.replay_buffer.sample(batch_size)
                
                # 将采样的经验添加到DDPG的回放缓冲区
                for state, action, reward, next_state, done in experiences:
                    self.model.replay_buffer.add(state, action, next_state, reward, done)
                
                # 训练模型
                self.model.train(gradient_steps=10, batch_size=64)
                
                # 每100轮保存模型
                if round_num % 100 == 0:
                    model_path = os.path.join(self.model_dir, f"ddpg_model_round_{round_num}.zip")
                    self.model.save(model_path)
                    print(f"Model saved to {model_path}")
                    
                    # 绘制训练曲线
                    self.monitor.plot_training_curves(
                        output_file=os.path.join(self.monitor.output_dir, f"training_curves_round_{round_num}.png")
                    )
            except Exception as e:
                print(f"Warning: Failed to train DDPG model: {e}")
                print(f"Continuing without training for this round.")
        
        # 重置当前状态和动作
        self.current_state = None
        self.current_action = None
        self.current_perceived_value = None
    
    def calculate_reward(self, result: Dict, profit: float) -> float:
        """
        计算复合奖励函数。
        """
        # 1. 直接利润奖励
        profit_reward = profit if profit is not None else 0.0
        
        # 2. 预算节奏奖励
        current_round = len(self.history)
        ideal_spend_per_round = self.initial_budget / self.total_rounds
        
        # 获取当前轮次的花费
        current_cost = self.history[-1]['cost'] if self.history else 0.0
        
        # 计算超支情况
        overspend = current_cost - ideal_spend_per_round
        pacing_reward = -max(0, overspend)
        
        # 3. 成本效率奖励
        if current_cost > 1e-8 and profit is not None and profit > 0:  # 避免除以零
            efficiency_reward = profit / (current_cost + 1e-8)
        else:
            efficiency_reward = 0.0
        
        # 组合奖励
        total_reward = (
            self.w_profit * profit_reward + 
            self.w_pacing * pacing_reward + 
            self.w_efficiency * efficiency_reward
        )
        
        return total_reward