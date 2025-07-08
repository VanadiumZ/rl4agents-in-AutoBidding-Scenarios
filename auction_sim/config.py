import numpy as np

# --- 模拟环境核心设置 ---
SIMULATION_ROUNDS = 1000  # 总共进行的拍卖轮次
N_SLOTS = 2  # 广告位的数量
CTR_POSITIONS = np.array([0.7, 0.3])  # 各个广告位的平均点击率
CTR_NOISE_STD = 0.05  # CTR的噪声标准差 (模拟±5%的扰动)

# --- 价值与预算设置 ---
TRUE_VALUE_RANGE = (0.5, 2.5)  # 真实价值V的均匀分布范围
AGENT_BUDGET = 100.0  # 智能体的初始预算
AGENT_PERCEPTION_NOISE_STD = 0.1  # 智能体对价值感知噪声的标准差 (ε)

# --- 规则智能体参数 ---
# “保守派”智能体平滑因子
CONSERVATIVE_AGENT_SMOOTHING = 0.8  # 用于平滑alpha变化的因子，防止调整过猛

# “激进派”智能体参数
AGGRESSIVE_AGENT_LOOKBACK = 15  # 回溯最近N轮的胜率
AGGRESSIVE_AGENT_LAMBDA = 0.5  # 调整出价的敏感度 λ

"""
--- 实验配置 ---
下面你可以根据这个配置来动态创建智能体
这是一个 k=0 的示例配置
格式: {'type': 'AgentType', 'count': N, 'budget': B}
Type可以是 'Truthful', 'Conservative', 'Aggressive', 'Learning'
"""

EXPERIMENT_SETUP = {
    'k': 0,
    'agents': [
        {'type': 'Conservative', 'count': 2, 'budget': 100.0},
        {'type': 'Aggressive', 'count': 2, 'budget': 120.0},
        # {'type': 'Truthful', 'count': 2, 'budget': 100.0}, # 可以任意组合
    ]
}