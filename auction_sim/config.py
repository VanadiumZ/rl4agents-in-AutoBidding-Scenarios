# /auction_sim/config.py
import numpy as np

# --- 模拟环境核心设置 ---
SIMULATION_ROUNDS = 16000  # 总共进行的拍卖轮次
N_SLOTS = 3  # 广告位的数量
CTR_POSITIONS = np.array([0.75, 0.4, 0.2])  # 各个广告位的平均点击率
# N_SLOT = 2 -> CTR_POSITIONS = np.array([0.75, 0.4])
CTR_NOISE_STD = 0.05  # CTR的噪声标准差 (模拟±5%的扰动)

# --- 价值与预算设置 ---
TRUE_VALUE_RANGE = (1.0, 20.0)  # 增加真实价值范围，提高收益潜力
AGENT_BUDGET = 24000.0  # 智能体的初始预算
AGENT_PERCEPTION_NOISE_STD = 0.2  # 降低感知噪声，减少出价偏差

# --- 规则智能体参数 ---
# "保守派"智能体平滑因子
CONSERVATIVE_AGENT_SMOOTHING = 0.8  # 用于平滑alpha变化的因子，防止调整过猛

# "激进派"智能体参数
AGGRESSIVE_AGENT_LOOKBACK = 15  # 回溯最近N轮的胜率
AGGRESSIVE_AGENT_LAMBDA = 0.5  # 调整出价的敏感度 λ

# --- 学习智能体参数 ---
# 强化学习算法选择: 'PPO' 或 'DDPG'
RL_ALGORITHM = 'PPO'  # 默认使用PPO算法

"""
--- 实验配置 ---
下面你可以根据这个配置来动态创建智能体
这是一个 k=0 的示例配置
格式: {'type': 'AgentType', 'count': N, 'budget': B}
Type可以是 'Truthful', 'Conservative', 'Aggressive', 'Learning'
"""

EXPERIMENT_SETUP = {
    'k': 1,
    'agents': [
        {'type': 'Conservative', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Aggressive', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Truthful', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Learning', 'count': 1, 'budget': AGENT_BUDGET},
    ]
}