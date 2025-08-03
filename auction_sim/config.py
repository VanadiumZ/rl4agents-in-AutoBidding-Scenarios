import numpy as np
from enum import Enum

# --- Curriculum Learning Stages ---
class CurriculumStage(Enum):
    SOLO = 0        # Solo practice - no competition
    GENTLE = 1      # Gentle competition - only Truthful agents
    MIXED = 2       # Mixed competition - Truthful + Conservative
    FULL = 3        # Full competition - all agent types

# --- 模拟环境核心设置 ---
SIMULATION_ROUNDS = 16000  # 总共进行的拍卖轮次
TRAINING_EPISODES = 200  # Increased from 50 for better convergence
N_SLOTS = 2  # 广告位的数量
# CTR_POSITIONS = np.array([0.7, 0.3])  # 原始设置：过低导致盈利困难
CTR_POSITIONS = np.array([0.95, 0.6])  # 改进：顶部位置95%，次位60% - 更接近真实广告场景
CTR_NOISE_STD = 0.05  # CTR的噪声标准差 (模拟±5%的扰动)

# --- 价值与预算设置 ---
# TRUE_VALUE_RANGE = (0.5, 10.0)  # 原始设置：价值范围较小，利润空间有限
TRUE_VALUE_RANGE = (5.0, 25.0)  # 改进：扩大价值范围，提供更多盈利机会
AGENT_BUDGET = 30000.0  # 智能体的初始预算
# 可以设置为[20000, 30000, 40000]，分别对应三种不同的实验
# AGENT_PERCEPTION_NOISE_STD = 0.3  # 原始设置：30%的噪声过高
AGENT_PERCEPTION_NOISE_STD = 0.2  # 改进：降低到20%，减少价值感知误差

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

# k=0: 规则智能体对照实验
EXPERIMENT_SETUP_K0 = {
    'k': 0,
    'agents': [
        {'type': 'Conservative', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Aggressive', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Truthful', 'count': 2, 'budget': AGENT_BUDGET},
    ]
}

# k=1: 单智能体强化学习 (1个学习智能体 + 规则对手)
EXPERIMENT_SETUP_K1 = {
    'k': 1,
    'agents': [
        {'type': 'Learning', 'count': 1, 'budget': AGENT_BUDGET},
        {'type': 'Conservative', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Aggressive', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Truthful', 'count': 2, 'budget': AGENT_BUDGET},
    ]
}

# k=2: 多智能体强化学习 (2个学习智能体 + 规则对手)
EXPERIMENT_SETUP_K2 = {
    'k': 2,
    'agents': [
        {'type': 'Learning', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Conservative', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Aggressive', 'count': 2, 'budget': AGENT_BUDGET},
        {'type': 'Truthful', 'count': 2, 'budget': AGENT_BUDGET},
    ]
}

# 当前实验设置 (可以切换 K0/K1/K2)
EXPERIMENT_SETUP = EXPERIMENT_SETUP_K2  # 切换这里来改变实验类型

# --- Symmetric Curriculum Learning Configurations ---
CURRICULUM_CONFIGS = {
    CurriculumStage.SOLO: {  # Now "COOPERATIVE" - both agents learn together vs easy opponents
        'n_learning': 2,  # SYMMETRIC: Both agents start together
        'n_truthful': 2,
        'n_conservative': 0,
        'n_aggressive': 0,
        'n_slots': 2,
        'budget': 40000,  # Generous budget for learning
        'max_rounds': 8000,
        'ctr_positions': np.array([0.95, 0.6]),  # 高CTR以支持初期学习
        'reward_weights': {
            'win_bonus': 0.6,      # High win bonus to encourage participation
            'profit': 0.3,         # Lower profit weight initially
            'cooperation': 0.1     # Small bonus for both agents winning
        }
    },
    CurriculumStage.GENTLE: {  # "COMPETITIVE" - add Conservative opponents
        'n_learning': 2,
        'n_truthful': 2, 
        'n_conservative': 2,
        'n_aggressive': 0,
        'n_slots': 2,
        'budget': 35000,
        'max_rounds': 12000,
        'ctr_positions': np.array([0.9, 0.55]),  # 略微降低CTR增加挑战
        'reward_weights': {
            'profit': 0.5,         # Increase profit importance
            'roi': 0.2,            # Introduce efficiency awareness
            'win_bonus': 0.3       # Reduce win bonus
        }
    },
    CurriculumStage.MIXED: {  # "ADVANCED" - add Aggressive opponents
        'n_learning': 2,
        'n_truthful': 2,
        'n_conservative': 2,
        'n_aggressive': 1,  # Start with just one aggressive
        'n_slots': 2,
        'budget': 32000,
        'max_rounds': 14000,
        'ctr_positions': np.array([0.85, 0.5]),  # 进一步平衡盈利与竞争
        'reward_weights': {
            'profit': 0.4,
            'roi': 0.3,
            'win_bonus': 0.2,
            'competition': 0.1     # Bonus for beating aggressive opponents
        }
    },
    CurriculumStage.FULL: {  # "FULL_SPECTRUM" - complete environment
        'n_learning': 2,
        'n_truthful': 2,
        'n_conservative': 2,
        'n_aggressive': 2,
        'n_slots': 2,
        'budget': 30000,
        'max_rounds': 16000,
        'ctr_positions': np.array([0.8, 0.45]),  # 最终阶段的平衡CTR
        'reward_transition': 'smooth',  # Gradual transition to final objective
        'final_objective_weights': {
            'profit': 0.5,
            'roi_scaled': 0.15,    # 0.15 * ROI * TotalCost
            'win_rate_scaled': 0.35  # 0.35 * WinRate * TargetWins
        }
    }
}

# Symmetric Curriculum success criteria
CURRICULUM_SUCCESS_CRITERIA = {
    CurriculumStage.SOLO: {  # COOPERATIVE stage
        'min_win_rate': 0.35,       # 35% for 4 agents (realistic expectation)
        'min_roi': -15.0,           # Allow negative ROI initially (learning phase)
        'min_budget_usage': 0.25,   # 25% budget usage (more realistic)
        'min_episodes': 30,         # Reduce episodes needed for faster progression
        'cooperation_bonus_threshold': 0.2  # Both agents win together 20% of time
    },
    CurriculumStage.GENTLE: {  # COMPETITIVE stage  
        'min_win_rate': 0.20,       # 20% for 6 agents (more realistic)
        'min_roi': 0.0,             # Break even
        'min_budget_usage': 0.40,   # 40% budget usage
        'min_episodes': 50
    },
    CurriculumStage.MIXED: {  # ADVANCED stage
        'min_win_rate': 0.18,       # 18% for 7 agents (2 learning + 5 opponents)
        'min_roi': 5.0,             # Positive ROI against harder opponents
        'min_budget_usage': 0.50,   # 50% budget usage
        'min_episodes': 75
    },
    CurriculumStage.FULL: {  # FULL_SPECTRUM stage
        'min_win_rate': 0.15,       # 15% for 8 agents (2 learning + 6 opponents)  
        'min_roi': 10.0,            # Good ROI in full competition
        'min_budget_usage': 0.60,   # 60% budget usage
        'min_episodes': 100,
        'economic_value_threshold': 0.6  # Top 60% in economic value ranking
    }
}
