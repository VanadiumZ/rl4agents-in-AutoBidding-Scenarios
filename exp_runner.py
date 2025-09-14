# /auction_sim/experiment_runner.py
import numpy as np
import random
from tqdm import tqdm
from typing import List, Dict, Any
from auction_sim import config
from auction_sim.auction import GSPAuction
from auction_sim.agents import Agent, TruthfulAgent, ConservativeAgent, AggressiveAgent, MultiAgentLearningAgent
from auction_sim.utils import generate_all_visualizations
from auction_sim.ippo_trainer import IPPOTrainer, IPPOActorCriticNetwork # 导入 IPPO 相关模块

def create_agents_from_config(total_rounds: int, model_paths: Dict[str, str] = None) -> List[Agent]:
    """根据config中的设置创建智能体列表，并加载预训练模型（如果提供了路径）"""
    agents = []
    agent_id_counter = 0
    learning_idx = 0  # 只对学习体计数，用于正确对齐模型
    total_agents = sum(spec['count'] for spec in config.EXPERIMENT_SETUP['agents'])
    learning_agents_spec = next((spec for spec in config.EXPERIMENT_SETUP['agents'] if spec['type'] == 'Learning'), None)
    
    # 初始化一个 IPPOTrainer，用于加载模型
    trainer = None
    if learning_agents_spec and model_paths:
        num_learning_agents = learning_agents_spec['count']
        trainer = IPPOTrainer(n_agents=num_learning_agents, obs_dim=7, action_dim=1)
        trainer.load_models(model_paths['learning_agent'])

    for spec in config.EXPERIMENT_SETUP['agents']:
        for _ in range(spec['count']):
            agent_id_prefix = spec['type']
            agent_id = f"{agent_id_prefix}_{agent_id_counter}"
            
            if agent_id_prefix == 'Truthful':
                agent = TruthfulAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD)
            elif agent_id_prefix == 'Conservative':
                agent = ConservativeAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, total_rounds)
            elif agent_id_prefix == 'Aggressive':
                agent = AggressiveAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, total_agents)
            elif agent_id_prefix == 'Learning':
                agent = MultiAgentLearningAgent(
                    agent_id=agent_id,
                    budget=spec['budget'],
                    perception_noise_std=config.AGENT_PERCEPTION_NOISE_STD,
                    is_training=False # 模拟时设置为非训练模式
                )
                # 将加载的模型与智能体关联，使用learning_idx正确对齐
                if trainer:
                    trainer_key = f"Learning_{learning_idx}"
                    if trainer_key in trainer.networks:
                        # 创建模型包装器以适配MultiAgentLearningAgent的接口
                        class ModelWrapper:
                            def __init__(self, network, trainer, key):
                                self.network = network
                                self.trainer = trainer
                                self.key = key
                            
                            def predict(self, obs, deterministic=True):
                                action, log_prob, value = self.trainer.get_action(self.key, obs, deterministic)
                                return np.array([action]), None
                        
                        model_wrapper = ModelWrapper(trainer.networks[trainer_key], trainer, trainer_key)
                        agent.set_model(model_wrapper)
                        print(f"Loaded trained model for {agent.id} <- {trainer_key}")
                learning_idx += 1  # 只对学习体递增
            else:
                raise ValueError(f"Unknown agent type: {agent_id_prefix}")
            
            agents.append(agent)
            agent_id_counter += 1
    
    print(f"Created {len(agents)} agents for the experiment.")
    return agents

def run_experiment(model_paths: Dict[str, str] = None, show_visuals: bool = True):
    """
    运行完整的拍卖模拟实验，并记录所有数据。

    Args:
        model_paths: 可选的字典，指定要加载的预训练模型路径。
                     例如：{'learning_agent': 'auction_sim/models/bc_ippo/final'}
        show_visuals: 是否生成并显示可视化图表。
    """
    # 1. 初始化
    random.seed(42)
    np.random.seed(42)
    
    auction = GSPAuction(config.N_SLOTS, config.CTR_POSITIONS, config.CTR_NOISE_STD)
    agents = create_agents_from_config(config.SIMULATION_ROUNDS, model_paths)
    
    # 存储所有回合数据
    all_round_data = []

    # 2. 运行模拟
    print("Starting simulation...")
    for round_num in tqdm(range(1, config.SIMULATION_ROUNDS + 1), desc="Auction Rounds"):
        true_value = random.uniform(*config.TRUE_VALUE_RANGE)
        
        bids = {}
        perceived_values = {}
        opponent_win_rates = {}
        
        # 收集对手信息，用于学习智能体的观测
        learning_agent_ids = [a.id for a in agents if isinstance(a, MultiAgentLearningAgent)]
        for agent in agents:
            if agent.id not in learning_agent_ids:
                if hasattr(agent, 'win_history') and agent.win_history:
                    opponent_win_rates[agent.id] = sum(agent.win_history) / len(agent.win_history)

        for agent in agents:
            perceived_value = agent.perceive(true_value)
            
            # 学习智能体需要额外的环境信息
            if isinstance(agent, MultiAgentLearningAgent):
                bid_price = agent.bid(
                    perceived_value,
                    current_round=round_num,
                    max_rounds=config.SIMULATION_ROUNDS,
                    opponent_win_rates=opponent_win_rates
                )
            else:
                bid_price = agent.bid(perceived_value)
            
            if agent.budget > 0:
                perceived_values[agent.id] = perceived_value
                bids[agent.id] = bid_price

        auction_results = auction.run_auction(bids) if bids else {}

        # 记录本回合数据
        round_data = {
            'round_num': round_num,
            'true_value': true_value,
            'bids': bids,
            'perceived_values': perceived_values,
            'results': auction_results,
            'profits': {},
            'costs': {}
        }
        
        for agent in agents:
            result = auction_results.get(agent.id)
            profit = 0.0
            if result and result['won']:
                true_value_profit = true_value * result['slot_ctr']
                expected_cost = result['cost_per_click'] * result['slot_ctr']
                cost = min(expected_cost, agent.budget)
                profit = true_value_profit - cost
                
            agent.update(result, round_num, true_value=true_value, profit=profit)
            
            round_data['profits'][agent.id] = profit
            round_data['costs'][agent.id] = agent.history[-1]['cost']
            
        all_round_data.append(round_data)

    # 3. 结果分析与展示
    print("\n--- Simulation Finished ---")
    print("Final Results:")
    print(f"{'Agent ID':<20} | {'Budget Left':<12} | {'Total Cost':<12} | {'Win Count':<10} | {'Cumulative Profit':<18} | {'ROI (%)':<10} | {'Budget Usage (%)':<18}")
    print("-" * 130)
    
    for agent in agents:
        # Convert numpy arrays to scalars using .item() method to avoid deprecation warnings
        initial_budget = agent.initial_budget.item() if hasattr(agent.initial_budget, 'item') else float(agent.initial_budget)
        total_cost = agent.get_total_cost().item() if hasattr(agent.get_total_cost(), 'item') else float(agent.get_total_cost())
        win_count = sum(1 for record in agent.history if record['result'] and record['result']['won'])
        cumulative_profit = agent.get_cumulative_profit().item() if hasattr(agent.get_cumulative_profit(), 'item') else float(agent.get_cumulative_profit())
        roi = agent.get_roi().item() if hasattr(agent.get_roi(), 'item') else float(agent.get_roi())
        current_budget = agent.budget.item() if hasattr(agent.budget, 'item') else float(agent.budget)
        budget_usage_raw = (initial_budget - current_budget) / initial_budget * 100
        budget_usage = float(budget_usage_raw) if hasattr(budget_usage_raw, 'item') else float(budget_usage_raw)
        
        # print(initial_budget, total_cost, win_count, cumulative_profit, roi, budget_usage)
        agent_id_str = str(agent.id)
        # Ensure agent_id_str is a proper string for formatting
        if hasattr(agent.id, '__iter__') and not isinstance(agent.id, str):
            agent_id_str = str(agent.id)
        else:
            agent_id_str = agent.id
        
        print(
            f"{agent_id_str:<20} | "
            f"{current_budget:8.2f}     | "
            f"{total_cost:8.2f}     | "
            f"{win_count:4d}       | "
            f"{cumulative_profit:8.2f}          | "
            f"{roi:8.2f}       | "
            f"{budget_usage:8.2f}"
        )

    # 4. 生成可视化图表
    if show_visuals:
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS...")
        print("="*50)
        
        try:
            generate_all_visualizations(agents, all_round_data)
            print("\n✅ All visualizations generated successfully!")
        except Exception as e:
            print(f"\n❌ Error generating visualizations: {e}")

    return agents, all_round_data

if __name__ == '__main__':
    # 示例：运行一个包含预训练学习智能体的实验
    # 请确保 'auction_sim/models/bc_ippo/final' 路径存在
    # 如果你没有预训练模型，请注释掉 model_paths 参数
    trained_model_path = 'auction_sim/models/bc_ippo/final'
    # trained_model_path = 'auction_sim\models\pure_curriculum\STAGE_8_final'
    # trained_model_path = 'auction_sim\models\ddpg'


    # try:
    # 尝试加载模型
    import os
    if not os.path.exists(trained_model_path):
        print(f"警告: 找不到模型路径 '{trained_model_path}'。将运行没有预训练模型的模拟。")
        final_agents, final_data = run_experiment(model_paths=None)
    else:
        final_agents, final_data = run_experiment(model_paths={'learning_agent': trained_model_path})
            
    # except Exception as e:
    #     print(f"运行失败: {e}")
    #     print("请检查你的配置和文件路径。")