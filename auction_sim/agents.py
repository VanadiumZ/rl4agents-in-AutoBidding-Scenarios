# /auction_sim/agents.py
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict
from collections import deque
from . import config

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
    """“自适应”学习智能体：使用强化学习模型进行出价"""
    def __init__(self, agent_id: str, budget: float, perception_noise_std: float):
        super().__init__(agent_id, budget, perception_noise_std)
        # TODO: 在这里初始化你的RL模型 (e.g., PPO from stable-baselines3)
        # self.model = PPO(...)
        # self.replay_buffer = ReplayBuffer(...)
        print(f"LearningAgent {self.id} initialized.")
    
    def get_state(self) -> np.ndarray:
        """
        构建并返回当前的状态向量，用于输入RL模型。
        这是你需要根据方案详细实现的部分。
        """
        # 这是一个示例状态，你需要填充真实数据
        state = np.array([
            0.0, # perceived_value_t
            0.0, # remaining_budget_ratio
            0.0, # time_ratio
            0.0, # recent_win_rate
            0.0, # recent_avg_profit_per_win
        ])
        return state

    def bid(self, perceived_value: float) -> float:
        """
        使用RL模型预测动作并转化为出价。
        """
        # 1. 获取当前状态
        # state = self.get_state(perceived_value, ...)
        
        # 2. 使用模型预测动作
        # action, _ = self.model.predict(state, deterministic=True)
        # raw_action = action[0] # PPO/DDPG通常返回一个数组
        
        # 3. 将动作缩放到合理的出价范围
        # bid_price = self.scale_action(raw_action, perceived_value)
        
        # [占位符] 目前使用随机出价作为示例
        bid_price = perceived_value * np.random.uniform(0.8, 1.2)
        
        return bid_price

    def scale_action(self, raw_action: float, perceived_value: float) -> float:
        """将模型输出 (-1, 1) 映射到出价。这是关键步骤！"""
        # 示例：将tanh输出映射到 [0, 2 * perceived_value]
        # max_bid_multiplier = 2.0
        # bid_price = (raw_action + 1) / 2 * max_bid_multiplier * perceived_value
        # return bid_price
        pass
    
    def update(self, result: Dict, round_num: int, true_value: float = None, profit: float = None):
        """
        更新智能体状态，并将 (s, a, r, s') 存入经验池。
        """
        super().update(result, round_num, true_value, profit)
        # TODO:
        # 1. 计算奖励 R
        # 2. 获取下一个状态 S'
        # 3. 将 (S, A, R, S') 添加到回放池
        # 4. (可选) 在这里或在runner中调用 model.learn()