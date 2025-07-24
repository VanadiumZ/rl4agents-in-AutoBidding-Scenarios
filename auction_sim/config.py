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
# “保守派”智能体平滑因子
CONSERVATIVE_AGENT_SMOOTHING = 0.8  # 用于平滑alpha变化的因子，防止调整过猛

# “激进派”智能体参数
AGGRESSIVE_AGENT_LOOKBACK = 15  # 回溯最近N轮的胜率
AGGRESSIVE_AGENT_LAMBDA = 0.5  # 调整出价的敏感度 λ

"""
--- 实验配置 ---
下面你可以根据这个配置来动态创建智能体
支持 k=0 (规则智能体), k=1 (单智能体RL), k=2 (多智能体RL)
格式: {'type': 'AgentType', 'count': N, 'budget': B}
Type可以是 'Truthful', 'Conservative', 'Aggressive', 'Learning'
"""

# k=0: 规则智能体对照实验
EXPERIMENT_SETUP_K0 = {
    'k': 0,
    'agents': [
        {'type': 'Conservative', 'count': 1, 'budget': AGENT_BUDGET},
        {'type': 'Aggressive', 'count': 1, 'budget': AGENT_BUDGET},
        {'type': 'Truthful', 'count': 1, 'budget': AGENT_BUDGET},
    ]
}

# k=1: 单智能体强化学习 (1个学习智能体 + 规则对手)
EXPERIMENT_SETUP_K1 = {
    'k': 1,
    'agents': [
        {'type': 'Learning', 'count': 1, 'budget': AGENT_BUDGET},
        {'type': 'Conservative', 'count': 1, 'budget': AGENT_BUDGET},
        {'type': 'Aggressive', 'count': 1, 'budget': AGENT_BUDGET},
        {'type': 'Truthful', 'count': 1, 'budget': AGENT_BUDGET},
    ]
}

# k=2: 多智能体强化学习 (2个学习智能体 + 规则对手)
EXPERIMENT_SETUP_K2 = {
    'k': 2,
    'agents': [
        {'type': 'Learning', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Conservative', 'count': 1, 'budget': AGENT_BUDGET},
        {'type': 'Aggressive', 'count': 1, 'budget': AGENT_BUDGET},
        {'type': 'Truthful', 'count': 1, 'budget': AGENT_BUDGET},
    ]
}

# 当前实验设置 (可以切换 K0/K1/K2)
EXPERIMENT_SETUP = EXPERIMENT_SETUP_K2  # 切换这里来改变实验类型

# 多智能体专用设置
MULTI_AGENT_TRAINING = {
    'n_episodes': 300,           # 训练轮数
    'max_steps_per_episode': SIMULATION_ROUNDS,  # 每轮最大步数
    'learning_rate': 3e-4,       # 学习率
    'gamma': 0.99,               # 折扣因子
    'gae_lambda': 0.95,          # GAE参数
    'clip_ratio': 0.2,           # PPO裁剪比率
    'update_epochs': 4,          # 每次更新的轮数
    'batch_size': 64,            # 批量大小
    'save_interval': 50,         # 模型保存间隔
    'eval_interval': 100,        # 评估间隔
}