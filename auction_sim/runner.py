# /auction_sim/runner.py
import numpy as np
import random
import os
from tqdm import tqdm
from . import config
from .auction import GSPAuction
from .agents import Agent, TruthfulAgent, ConservativeAgent, AggressiveAgent, LearningAgent, DDPGLearningAgent
from .training_monitor import TrainingMonitor

def create_agents_from_config():
    """根据config中的设置创建智能体列表"""
    agents = []
    agent_id_counter = 0
    total_agents = sum(spec['count'] for spec in config.EXPERIMENT_SETUP['agents'])
    
    # 创建模型和监控器保存目录
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    for spec in config.EXPERIMENT_SETUP['agents']:
        for _ in range(spec['count']):
            agent_id = f"{spec['type']}_{agent_id_counter}"
            agent_id_counter += 1
            
            if spec['type'] == 'Truthful':
                agents.append(TruthfulAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD))
            elif spec['type'] == 'Conservative':
                agents.append(ConservativeAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, config.SIMULATION_ROUNDS))
            elif spec['type'] == 'Aggressive':
                agents.append(AggressiveAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, total_agents))
            elif spec['type'] == 'Learning':
                # k > 0 时会创建学习智能体
                # 创建训练监控器
                algorithm = 'DDPG' if hasattr(config, 'RL_ALGORITHM') and config.RL_ALGORITHM == 'DDPG' else 'PPO'
                monitor = TrainingMonitor(
                    agent_id=agent_id,
                    algorithm=algorithm,
                    log_dir=os.path.join(logs_dir, f"{agent_id}_{algorithm}"),
                    model_dir=os.path.join(models_dir, f"{agent_id}_{algorithm}")
                )
                
                # 根据配置选择使用PPO或DDPG算法
                if hasattr(config, 'RL_ALGORITHM') and config.RL_ALGORITHM == 'DDPG':
                    agents.append(DDPGLearningAgent(
                        agent_id, 
                        spec['budget'], 
                        config.AGENT_PERCEPTION_NOISE_STD,
                        monitor=monitor
                    ))
                    print(f"Created DDPGLearningAgent: {agent_id} with TrainingMonitor")
                else:
                    agents.append(LearningAgent(
                        agent_id, 
                        spec['budget'], 
                        config.AGENT_PERCEPTION_NOISE_STD,
                        monitor=monitor
                    ))
                    print(f"Created PPO LearningAgent: {agent_id} with TrainingMonitor")
            else:
                raise ValueError(f"Unknown agent type: {spec['type']}")
    
    print(f"Created {len(agents)} agents for the experiment.")
    return agents


def main():
    """主模拟函数"""
    # 1. 初始化
    random.seed(42)
    np.random.seed(42)
    
    auction = GSPAuction(config.N_SLOTS, config.CTR_POSITIONS, config.CTR_NOISE_STD)
    agents = create_agents_from_config()
    
    all_history = {agent.id: [] for agent in agents}

    # 2. 运行模拟
    print("Starting simulation...")
    for round_num in tqdm(range(1, config.SIMULATION_ROUNDS + 1), desc="Auction Rounds"):
        # 1. 生成真实价值
        true_value = random.uniform(*config.TRUE_VALUE_RANGE)
        
        # 2. 所有智能体出价
        bids = {}
        perceived_values = {}
        for agent in agents:
            perceived_value = agent.perceive(true_value)
            bid_price = agent.bid(perceived_value)
            
            # 只有当智能体有足够预算时才参与竞价
            if agent.can_afford_bid(bid_price):
                perceived_values[agent.id] = perceived_value
                bids[agent.id] = bid_price

        # 3. 运行拍卖
        if bids:
            auction_results = auction.run_auction(bids)
        else:
            auction_results = {}  # 本轮无人出价

        # 4. 更新所有智能体状态并计算利润
        for agent in agents:
            result = auction_results.get(agent.id)
            
            # 计算利润：Profit_t = TrueValue_t × CTR_pos - Cost_t
            if result and result['won']:
                true_value_profit = true_value * result['slot_ctr']
                expected_cost = result['cost_per_click'] * result['slot_ctr']
                profit = true_value_profit - expected_cost
            else:
                profit = 0.0
            
            # 更新智能体状态，传递真实价值和利润信息
            agent.update(result, round_num, true_value=true_value, profit=profit)

    # 3. 保存训练监控器数据并生成训练曲线
    learning_agents = [agent for agent in agents if isinstance(agent, (LearningAgent, DDPGLearningAgent))]
    if learning_agents:
        print("\n--- Saving Training Data and Generating Plots ---")
        for agent in learning_agents:
            if hasattr(agent, 'monitor') and agent.monitor is not None:
                # 保存训练数据
                agent.monitor.save_metrics()
                # 生成训练曲线
                agent.monitor.plot_training_curves()
                print(f"Training data and plots saved for {agent.id}")
    
    # 4. 结果分析
    print("\n--- Simulation Finished ---")
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

if __name__ == '__main__':
    main()