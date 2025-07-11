import numpy as np
import random
from tqdm import tqdm
from . import config
from .auction import GSPAuction
from .agents import Agent, TruthfulAgent, ConservativeAgent, AggressiveAgent, LearningAgent

def create_agents_from_config():
    """根据config中的设置创建智能体列表"""
    agents = []
    agent_id_counter = 0
    total_agents = sum(spec['count'] for spec in config.EXPERIMENT_SETUP['agents'])

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
                agents.append(LearningAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD))
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
            # 检查预算是否充足
            if agent.budget > 0:
                perceived_value = agent.perceive(true_value)
                bid_price = agent.bid(perceived_value)

                # 预算不足以覆盖本轮出价时选择弃权，以避免后续出现负预算
                if agent.budget >= bid_price:
                    perceived_values[agent.id] = perceived_value
                    bids[agent.id] = bid_price

        # 3. 运行拍卖
        if bids:
            auction_results = auction.run_auction(bids)
        else:
            auction_results = {}  # 本轮无人出价

        # 4. 更新所有智能体状态
        for agent in agents:
            result_for_agent = auction_results.get(agent.id)
            # update方法会处理输掉拍卖或未参与拍卖的情况 (result_for_agent is None)
            agent.update(result_for_agent, round_num)

    # 3. 结果分析 (简单打印)
    print("\n--- Simulation Finished ---")
    print("Final Results:")
    for agent in agents:
        total_cost = agent.initial_budget - agent.budget
        
        # 计算总利润 (基于感知价值)
        # 注意: 您的方案中Profit公式用的是TrueValue，这里为了简化，我们先用PerceivedValue
        # 最终版本应记录TrueValue来计算真实利润
        total_profit = 0
        win_count = 0
        for record in agent.history:
            res = record['result']
            if res and res['won']:
                win_count += 1
                # 这里的perceived_value需要从当轮记录中获取，为简化暂时不计算
                # profit_this_round = perceived_values[agent.id] * res['slot_ctr'] - res['cost_per_click']
                # total_profit += profit_this_round
        
        roi = 0
        if total_cost > 0:
            # roi = (total_profit / total_cost) * 100 # 等待实现Profit计算
            pass

        print(
            f"{agent.id:<20} | "
            f"Budget Left: {agent.budget:8.2f} | "
            f"Total Cost: {total_cost:8.2f} | "
            f"Win Count: {win_count:4d}"
        )

if __name__ == '__main__':
    main()