# /auction_sim/model_runner.py
"""
模型评估运行器 - 使用训练好的模型进行拍卖模拟
支持加载BC、PPO、MAPPO等训练好的模型
"""
import numpy as np
import random
import torch
import os
from tqdm import tqdm
from . import config
from .auction import GSPAuction
from .agents import Agent, TruthfulAgent, ConservativeAgent, AggressiveAgent, SingleAgentLearningAgent
from .utils import generate_all_visualizations
from .ma_trainer import MAPPOTrainer, ActorCriticNetwork
from .ppo_trainer import PPOTrainer
from .bc_trainer import BCTrainer

def load_trained_model(model_path: str, model_type: str = "auto"):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径或目录路径
        model_type: 模型类型 ("bc", "ppo", "mappo", "maddpg", "auto")
    
    Returns:
        加载好的模型对象
    """
    if model_type == "auto":
        # 自动检测模型类型
        if os.path.isdir(model_path):
            files_in_dir = os.listdir(model_path)
            if "shared_model.pth" in files_in_dir:
                model_type = "mappo"
            elif any(f.endswith("_ppo_model.pth") for f in files_in_dir):
                model_type = "ppo"
            elif any(f.endswith("_actor.pth") for f in files_in_dir) and any(f.endswith("_critic.pth") for f in files_in_dir):
                model_type = "maddpg"
            elif "Learning_0_model.pth" in files_in_dir:
                model_type = "bc"
            else:
                model_type = "bc"
        else:
            if "bc" in model_path.lower():
                model_type = "bc"
            elif "ppo" in model_path.lower():
                model_type = "ppo"
            elif "maddpg" in model_path.lower():
                model_type = "maddpg"
            else:
                model_type = "mappo"
    
    print(f"Loading {model_type.upper()} model from {model_path}")
    
    if model_type == "bc":
        # 加载BC模型
        trainer = BCTrainer()
        if os.path.isfile(model_path):
            # 直接加载文件
            trainer.network.load_state_dict(torch.load(model_path))
        else:
            raise FileNotFoundError(f"BC模型文件不存在: {model_path}")
        return trainer, "bc"
    
    elif model_type == "ppo":
        # 加载PPO模型
        trainer = PPOTrainer()
        trainer.load_models(model_path)
        return trainer, "ppo"
    
    elif model_type == "mappo":
        # 加载MAPPO模型
        trainer = MAPPOTrainer()
        trainer.load_models(model_path)
        return trainer, "mappo"
    
    elif model_type == "maddpg":
        # 加载MADDPG模型
        # 注意：这里需要根据实际的MADDPG trainer实现来调整
        print("MADDPG model loading not implemented yet")
        raise NotImplementedError("MADDPG model loading not implemented")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def create_agents_with_trained_model(model_trainer, model_type: str):
    """
    使用训练好的模型创建智能体
    
    Args:
        model_trainer: 加载好的模型训练器
        model_type: 模型类型
    
    Returns:
        智能体列表
    """
    agents = []
    agent_id_counter = 0
    
    for spec in config.EXPERIMENT_SETUP['agents']:
        for _ in range(spec['count']):
            agent_id = f"{spec['type']}_{agent_id_counter}"
            agent_id_counter += 1
            
            if spec['type'] == 'Truthful':
                agents.append(TruthfulAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD))
            elif spec['type'] == 'Conservative':
                agents.append(ConservativeAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, config.SIMULATION_ROUNDS))
            elif spec['type'] == 'Aggressive':
                total_agents = sum(s['count'] for s in config.EXPERIMENT_SETUP['agents'])
                agents.append(AggressiveAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, total_agents))
            elif spec['type'] == 'Learning':
                # 创建学习智能体并设置训练好的模型
                learning_agent = SingleAgentLearningAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD)
                
                # 根据模型类型创建模型包装器
                if model_type == "bc":
                    class BCModelWrapper:
                        def __init__(self, network):
                            self.network = network
                        
                        def predict(self, obs, deterministic=True):
                            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                            with torch.no_grad():
                                mean, _, _ = self.network(obs_tensor)
                                action = mean.cpu().numpy().flatten()[0]
                            return np.array([action]), None
                    
                    model_wrapper = BCModelWrapper(model_trainer.network)
                
                elif model_type == "ppo":
                    class PPOModelWrapper:
                        def __init__(self, trainer, agent_id):
                            self.trainer = trainer
                            self.agent_id = agent_id
                        
                        def predict(self, obs, deterministic=True):
                            action, _ = self.trainer.get_action(self.agent_id, obs, deterministic)
                            return np.array([action]), None
                    
                    model_wrapper = PPOModelWrapper(model_trainer, agent_id)
                
                elif model_type == "mappo":
                    class MAPPOModelWrapper:
                        def __init__(self, trainer, agent_id):
                            self.trainer = trainer
                            self.agent_id = agent_id
                            # 确保智能体存在于trainer中
                            if agent_id not in self.trainer.networks:
                                self.trainer.add_agent(agent_id)
                        
                        def predict(self, obs, deterministic=True):
                            action, _ = self.trainer.get_action(self.agent_id, obs, deterministic)
                            return np.array([action]), None
                    
                    model_wrapper = MAPPOModelWrapper(model_trainer, agent_id)
                
                learning_agent.set_model(model_wrapper)
                agents.append(learning_agent)
            else:
                raise ValueError(f"Unknown agent type: {spec['type']}")
    
    print(f"Created {len(agents)} agents with trained models.")
    return agents

def run_simulation_with_trained_model(model_path: str, model_type: str = "auto", n_rounds: int = None):
    """
    使用训练好的模型运行拍卖模拟
    
    Args:
        model_path: 模型文件路径
        model_type: 模型类型
        n_rounds: 模拟轮数，默认使用config中的设置
    """
    # 1. 加载训练好的模型
    model_trainer, detected_type = load_trained_model(model_path, model_type)
    print(f"Successfully loaded {detected_type.upper()} model")
    
    # 2. 初始化
    random.seed(42)
    np.random.seed(42)
    
    auction = GSPAuction(config.N_SLOTS, config.CTR_POSITIONS, config.CTR_NOISE_STD)
    agents = create_agents_with_trained_model(model_trainer, detected_type)
    
    simulation_rounds = n_rounds if n_rounds is not None else config.SIMULATION_ROUNDS
    all_history = {agent.id: [] for agent in agents}
    
    # 3. 运行模拟
    print(f"Starting simulation with trained {detected_type.upper()} model...")
    for round_num in tqdm(range(1, simulation_rounds + 1), desc="Auction Rounds"):
        # 生成真实价值
        true_value = random.uniform(*config.TRUE_VALUE_RANGE)
        
        # 所有智能体出价
        bids = {}
        perceived_values = {}
        
        # 计算对手胜率（用于Learning智能体的观察）
        opponent_win_rates = {}
        for agent in agents:
            if hasattr(agent, 'history') and len(agent.history) > 0:
                wins = sum(1 for record in agent.history if record.get('result') and record['result'].get('won', False))
                opponent_win_rates[agent.id] = wins / len(agent.history)
            else:
                opponent_win_rates[agent.id] = 0.0
        
        for agent in agents:
            perceived_value = agent.perceive(true_value)
            
            # 为SingleAgentLearningAgent传递完整参数
            if hasattr(agent, 'get_observation'):  # SingleAgentLearningAgent
                bid_price = agent.bid(perceived_value, round_num, simulation_rounds, opponent_win_rates)
            else:  # 其他智能体类型
                bid_price = agent.bid(perceived_value)
            
            # 只有当智能体有足够预算时才参与竞价
            if agent.can_afford_bid(bid_price):
                perceived_values[agent.id] = perceived_value
                bids[agent.id] = bid_price
        
        # 运行拍卖
        if bids:
            auction_results = auction.run_auction(bids)
        else:
            auction_results = {}  # 本轮无人出价
        
        # 更新所有智能体状态并计算利润
        for agent in agents:
            result = auction_results.get(agent.id)
            
            # 计算利润：Profit_t = TrueValue_t × CTR_pos - Cost_t
            if result and result['won']:
                true_value_profit = true_value * result['slot_ctr']
                expected_cost = result['cost_per_click'] * result['slot_ctr']
                profit = true_value_profit - expected_cost
            else:
                profit = 0.0
            
            # 更新智能体状态
            agent.update(result, round_num, true_value=true_value, profit=profit)
    
    # 4. 结果分析
    print("\n--- Simulation with Trained Model Finished ---")
    print(f"Model Type: {detected_type.upper()}")
    print(f"Model Path: {model_path}")
    print("Final Results:")
    print(f"{'Agent ID':<20} | {'Budget Left':<12} | {'Total Cost':<12} | {'Win Count':<10} | {'Cumulative Profit':<18} | {'ROI (%)':<10}")
    print("-" * 95)
    
    for agent in agents:
        total_cost = agent.get_total_cost()
        win_count = sum(1 for record in agent.history if record['result'] and record['result']['won'])
        cumulative_profit = agent.get_cumulative_profit()
        roi = agent.get_roi()
        
        print(
            f"{agent.id:<20} | "
            f"{agent.budget:8.2f}     | "
            f"{total_cost:8.2f}     | "
            f"{win_count:4d}       | "
            f"{cumulative_profit:8.2f}          | "
            f"{roi:8.2f}"
        )
    
    # 5. 生成可视化图表
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS...")
    print("="*50)
    
    try:
        generate_all_visualizations(agents)
        print("\n✅ All visualizations generated successfully!")
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        print("You may need to install matplotlib, pandas, and scipy:")
        print("pip install matplotlib pandas scipy")
    
    return agents

def main():
    """
    主函数 - 演示如何使用训练好的模型
    """
    print("Model Runner - 使用训练好的模型进行拍卖模拟")
    print("="*60)
    
    # 示例：加载不同类型的模型
    model_examples = [
        ("auction_sim/models/bc_model.pth", "bc"),
        ("auction_sim/models/best", "mappo"),
        ("auction_sim/models/ppo", "ppo")
    ]
    
    for model_path, model_type in model_examples:
        if os.path.exists(model_path):
            print(f"\n运行 {model_type.upper()} 模型模拟...")
            try:
                agents = run_simulation_with_trained_model(model_path, model_type)
                print(f"✅ {model_type.upper()} 模型模拟完成")
            except Exception as e:
                print(f"❌ {model_type.upper()} 模型模拟失败: {e}")
        else:
            print(f"⚠️  模型文件不存在: {model_path}")

if __name__ == '__main__':
    main()