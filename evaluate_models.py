#!/usr/bin/env python3
"""
模型评估脚本：加载已训练的模型并评估其性能
用于获取准确的性能指标，避免重新训练
"""

import numpy as np
import torch
import torch.nn as nn
import os
import sys
from collections import defaultdict
import json
from tqdm import tqdm

# Add auction_sim to path
sys.path.append('.')

from auction_sim import config
from auction_sim.config import CurriculumStage, CURRICULUM_CONFIGS
from auction_sim.ma_environment import MultiAgentAuctionEnv
from auction_sim.agents import TruthfulAgent, ConservativeAgent, AggressiveAgent, MultiAgentLearningAgent
from auction_sim.ma_trainer import ActorCriticNetwork
from auction_sim.ippo_trainer import IPPOActorCriticNetwork

class ModelEvaluator:
    """评估已训练模型的性能"""
    
    def __init__(self, n_eval_episodes=100):
        """
        Args:
            n_eval_episodes: 评估的episode数量
        """
        self.n_eval_episodes = n_eval_episodes
        self.stage = CurriculumStage.STAGE_8  # 使用完整8智能体环境
        self.stage_config = CURRICULUM_CONFIGS[self.stage]
        
    def create_rule_agents(self):
        """创建规则智能体"""
        rule_agents = []
        agent_id_counter = 0
        
        # Truthful agents
        for _ in range(self.stage_config.get('n_truthful', 0)):
            agent = TruthfulAgent(
                f"Truthful_{agent_id_counter}",
                self.stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD
            )
            rule_agents.append(agent)
            agent_id_counter += 1
            
        # Conservative agents
        for _ in range(self.stage_config.get('n_conservative', 0)):
            agent = ConservativeAgent(
                f"Conservative_{agent_id_counter}",
                self.stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD,
                self.stage_config['max_rounds']
            )
            rule_agents.append(agent)
            agent_id_counter += 1
            
        # Aggressive agents
        for _ in range(self.stage_config.get('n_aggressive', 0)):
            total_agents = sum([self.stage_config.get(f'n_{t}', 0) 
                               for t in ['learning', 'truthful', 'conservative', 'aggressive']])
            agent = AggressiveAgent(
                f"Aggressive_{agent_id_counter}",
                self.stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD,
                total_agents
            )
            rule_agents.append(agent)
            agent_id_counter += 1
            
        return rule_agents
    
    def evaluate_mappo(self, model_dir="auction_sim/models/bc_direct/final"):
        """评估MAPPO模型（注意：实际实现中每个agent有独立网络）"""
        print("\n" + "="*70)
        print("评估 BC + MAPPO (bc_direct) 模型")
        print("="*70)
        
        # 创建环境
        learning_agent_ids = [f"Learning_{i}" for i in range(2)]
        rule_agents = self.create_rule_agents()
        env = MultiAgentAuctionEnv(learning_agent_ids, rule_agents, self.stage)
        
        # 创建学习智能体，每个agent加载自己的模型
        learning_agents = []
        for agent_id in learning_agent_ids:
            model_path = f"{model_dir}/{agent_id}_model.pth"
            if not os.path.exists(model_path):
                # 尝试使用hybrid_best目录
                model_path = f"auction_sim/models/hybrid_best/{agent_id}_model.pth"
                if not os.path.exists(model_path):
                    print(f"❌ 模型文件不存在: {model_path}")
                    return None
            
            # 为每个agent创建独立的网络
            network = ActorCriticNetwork(obs_dim=7, action_dim=1)
            network.load_state_dict(torch.load(model_path))
            network.eval()
            
            agent = MultiAgentLearningAgent(
                agent_id=agent_id,
                budget=self.stage_config['budget'],
                perception_noise_std=config.AGENT_PERCEPTION_NOISE_STD,
                model=None,
                is_training=False
            )
            
            # 创建模型包装器
            class ModelWrapper:
                def __init__(self, network):
                    self.network = network
                    
                def predict(self, obs, deterministic=True):
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        action, _ = self.network.get_action(obs_tensor, deterministic=deterministic)
                        return np.array([action.item()]), None
            
            agent.set_model(ModelWrapper(network))
            learning_agents.append(agent)
        
        # 运行评估
        return self._run_evaluation(env, learning_agents, learning_agent_ids, "MAPPO")
    
    def evaluate_ippo(self, model_dir="auction_sim/models/bc_ippo/final"):
        """评估IPPO模型（独立网络）"""
        print("\n" + "="*70)
        print("评估 BC + IPPO 模型")
        print("="*70)
        
        # 创建环境
        learning_agent_ids = [f"Learning_{i}" for i in range(2)]
        rule_agents = self.create_rule_agents()
        env = MultiAgentAuctionEnv(learning_agent_ids, rule_agents, self.stage)
        
        # 创建学习智能体（每个有独立模型）
        learning_agents = []
        for i, agent_id in enumerate(learning_agent_ids):
            model_path = f"{model_dir}/{agent_id}_ippo_model.pth"
            if not os.path.exists(model_path):
                print(f"❌ 模型文件不存在: {model_path}")
                return None
                
            # 创建独立网络并加载权重
            network = IPPOActorCriticNetwork(obs_dim=7, action_dim=1)
            network.load_state_dict(torch.load(model_path))
            network.eval()
            
            agent = MultiAgentLearningAgent(
                agent_id=agent_id,
                budget=self.stage_config['budget'],
                perception_noise_std=config.AGENT_PERCEPTION_NOISE_STD,
                model=None,
                is_training=False
            )
            
            # 创建模型包装器
            class ModelWrapper:
                def __init__(self, network):
                    self.network = network
                    
                def predict(self, obs, deterministic=True):
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        action, _ = self.network.get_action(obs_tensor, deterministic=deterministic)
                        return np.array([action.item()]), None
            
            agent.set_model(ModelWrapper(network))
            learning_agents.append(agent)
        
        # 运行评估
        return self._run_evaluation(env, learning_agents, learning_agent_ids, "IPPO")
    
    def evaluate_maddpg(self, model_dir="auction_sim/models/bc_maddpg/final"):
        """评估MADDPG模型"""
        print("\n" + "="*70)
        print("评估 BC + MADDPG 模型")
        print("="*70)
        
        # 创建环境
        learning_agent_ids = [f"Learning_{i}" for i in range(2)]
        rule_agents = self.create_rule_agents()
        env = MultiAgentAuctionEnv(learning_agent_ids, rule_agents, self.stage)
        
        # 导入MADDPG的Actor网络
        from bc_maddpg_trainer import Actor
        
        # 创建学习智能体
        learning_agents = []
        for agent_id in learning_agent_ids:
            model_path = f"{model_dir}/{agent_id}_actor.pth"
            if not os.path.exists(model_path):
                print(f"❌ 模型文件不存在: {model_path}")
                return None
                
            # 创建Actor网络并加载权重
            actor = Actor(obs_dim=7, action_dim=1)
            actor.load_state_dict(torch.load(model_path))
            actor.eval()
            
            agent = MultiAgentLearningAgent(
                agent_id=agent_id,
                budget=self.stage_config['budget'],
                perception_noise_std=config.AGENT_PERCEPTION_NOISE_STD,
                model=None,
                is_training=False
            )
            
            # 创建模型包装器
            class ModelWrapper:
                def __init__(self, actor):
                    self.actor = actor
                    
                def predict(self, obs, deterministic=True):
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        action = self.actor(obs_tensor).squeeze(0).numpy()
                        # action已经在Actor的forward中缩放到[0.1, 2.0]
                        return np.array([action.item()]), None
            
            agent.set_model(ModelWrapper(actor))
            learning_agents.append(agent)
        
        # 运行评估
        return self._run_evaluation(env, learning_agents, learning_agent_ids, "MADDPG")
    
    def _run_evaluation(self, env, learning_agents, learning_agent_ids, algorithm_name):
        """运行评估episodes并收集统计数据"""
        print(f"运行 {self.n_eval_episodes} 个评估episodes...")
        
        # 统计数据收集
        individual_stats = {agent_id: {
            'wins': [],
            'profits': [],
            'costs': [],
            'rois': [],
            'budget_usage': []
        } for agent_id in learning_agent_ids}
        
        episode_stats = []
        
        for episode in tqdm(range(self.n_eval_episodes), desc=f"评估{algorithm_name}"):
            # 重置环境
            states = env.reset()
            
            # 重置每个episode的统计
            ep_stats = {agent_id: {
                'wins': 0,
                'profit': 0,
                'cost': 0,
                'steps': 0
            } for agent_id in learning_agent_ids}
            
            # 记录episode开始前的历史长度
            start_history_length = {
                agent_id: len(env.agent_histories[agent_id]) 
                for agent_id in learning_agent_ids
            }
            
            # 运行一个episode
            for step in range(self.stage_config['max_rounds']):
                # 获取动作
                actions = {}
                for i, agent in enumerate(learning_agents):
                    agent_id = learning_agent_ids[i]
                    action, _ = agent.model.predict(states[agent_id], deterministic=True)
                    actions[agent_id] = action
                
                # 执行动作
                step_result = env.step(actions)
                if len(step_result) == 5:
                    next_states, rewards, terminated, truncated, infos = step_result
                    dones = {k: terminated[k] or truncated[k] for k in terminated.keys()}
                else:
                    next_states, rewards, dones, infos = step_result
                
                states = next_states
                
                if any(dones.values()):
                    break
            
            # episode结束后，从agent_histories中提取数据
            for agent_id in learning_agent_ids:
                agent_history = env.agent_histories[agent_id]
                start_len = start_history_length[agent_id]
                episode_history = agent_history[start_len:]  # 只看这个episode的历史
                
                # 统计这个episode的数据
                ep_stats[agent_id]['steps'] = len(episode_history)
                ep_stats[agent_id]['wins'] = sum(1 for h in episode_history if h.get('won', False))
                ep_stats[agent_id]['profit'] = sum(h.get('profit', 0) for h in episode_history)
                ep_stats[agent_id]['cost'] = sum(h.get('cost', 0) for h in episode_history)
            
            # 计算episode统计
            for agent_id in learning_agent_ids:
                wins = ep_stats[agent_id]['wins']
                steps = ep_stats[agent_id]['steps']
                profit = ep_stats[agent_id]['profit']
                cost = ep_stats[agent_id]['cost']
                
                win_rate = wins / steps if steps > 0 else 0
                roi = (profit / cost * 100) if cost > 0 else 0
                
                initial_budget = self.stage_config['budget']
                remaining = env.agent_budgets.get(agent_id, initial_budget)
                budget_used = (initial_budget - remaining) / initial_budget
                
                individual_stats[agent_id]['wins'].append(win_rate)
                individual_stats[agent_id]['profits'].append(profit)
                individual_stats[agent_id]['costs'].append(cost)
                individual_stats[agent_id]['rois'].append(roi)
                individual_stats[agent_id]['budget_usage'].append(budget_used)
            
            episode_stats.append(ep_stats)
        
        # 计算最终统计
        results = {
            'algorithm': algorithm_name,
            'n_episodes': self.n_eval_episodes,
            'individual_performance': {}
        }
        
        for agent_id in learning_agent_ids:
            stats = individual_stats[agent_id]
            results['individual_performance'][agent_id] = {
                'avg_win_rate': np.mean(stats['wins']),
                'std_win_rate': np.std(stats['wins']),
                'avg_roi': np.mean(stats['rois']),
                'std_roi': np.std(stats['rois']),
                'avg_profit': np.mean(stats['profits']),
                'avg_cost': np.mean(stats['costs']),
                'avg_budget_usage': np.mean(stats['budget_usage'])
            }
        
        # 计算汇总统计
        all_win_rates = []
        all_rois = []
        for agent_id in learning_agent_ids:
            all_win_rates.extend(individual_stats[agent_id]['wins'])
            all_rois.extend(individual_stats[agent_id]['rois'])
        
        results['aggregate_stats'] = {
            'avg_win_rate': np.mean(all_win_rates),
            'avg_roi': np.mean(all_rois),
            'win_rate_difference': abs(
                results['individual_performance']['Learning_0']['avg_win_rate'] -
                results['individual_performance']['Learning_1']['avg_win_rate']
            ),
            'roi_difference': abs(
                results['individual_performance']['Learning_0']['avg_roi'] -
                results['individual_performance']['Learning_1']['avg_roi']
            )
        }
        
        # 打印结果
        self._print_results(results)
        
        return results
    
    def _print_results(self, results):
        """打印评估结果"""
        print(f"\n{results['algorithm']} 评估结果 ({results['n_episodes']} episodes):")
        print("-" * 50)
        
        # 个体表现
        for agent_id, stats in results['individual_performance'].items():
            print(f"\n{agent_id}:")
            print(f"  平均胜率: {stats['avg_win_rate']:.1%} (±{stats['std_win_rate']:.1%})")
            print(f"  平均ROI: {stats['avg_roi']:.1f}% (±{stats['std_roi']:.1f}%)")
            print(f"  平均利润: {stats['avg_profit']:.1f}")
            print(f"  平均成本: {stats['avg_cost']:.1f}")
            print(f"  预算使用: {stats['avg_budget_usage']:.1%}")
        
        # 汇总统计
        agg = results['aggregate_stats']
        print(f"\n汇总统计:")
        print(f"  总平均胜率: {agg['avg_win_rate']:.1%}")
        print(f"  总平均ROI: {agg['avg_roi']:.1f}%")
        print(f"  胜率差异: {agg['win_rate_difference']:.1%}")
        print(f"  ROI差异: {agg['roi_difference']:.1f}%")
    
    def evaluate_all(self):
        """评估所有三种算法"""
        all_results = {}
        
        # 评估MAPPO
        mappo_results = self.evaluate_mappo()
        if mappo_results:
            all_results['MAPPO'] = mappo_results
            
        # 评估IPPO
        ippo_results = self.evaluate_ippo()
        if ippo_results:
            all_results['IPPO'] = ippo_results
            
        # 评估MADDPG
        maddpg_results = self.evaluate_maddpg()
        if maddpg_results:
            all_results['MADDPG'] = maddpg_results
        
        # 保存结果
        if all_results:
            output_file = "evaluation_results.json"
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n结果已保存到: {output_file}")
        
        # 打印对比表
        self._print_comparison_table(all_results)
        
        return all_results
    
    def _print_comparison_table(self, all_results):
        """打印三种算法的对比表"""
        if not all_results:
            return
            
        print("\n" + "="*80)
        print("三种算法对比总结")
        print("="*80)
        
        # 表头
        print(f"{'算法':<12} {'L0胜率':>10} {'L1胜率':>10} {'平均胜率':>10} {'胜率差异':>10} "
              f"{'L0_ROI':>10} {'L1_ROI':>10} {'平均ROI':>10} {'ROI差异':>10}")
        print("-" * 102)
        
        # 数据行
        for algo_name, results in all_results.items():
            l0_stats = results['individual_performance']['Learning_0']
            l1_stats = results['individual_performance']['Learning_1']
            agg_stats = results['aggregate_stats']
            
            print(f"{algo_name:<12} "
                  f"{l0_stats['avg_win_rate']:>9.1%} "
                  f"{l1_stats['avg_win_rate']:>9.1%} "
                  f"{agg_stats['avg_win_rate']:>9.1%} "
                  f"{agg_stats['win_rate_difference']:>9.1%} "
                  f"{l0_stats['avg_roi']:>9.1f}% "
                  f"{l1_stats['avg_roi']:>9.1f}% "
                  f"{agg_stats['avg_roi']:>9.1f}% "
                  f"{agg_stats['roi_difference']:>9.1f}%")
        
        print("="*102)


def main():
    """主函数"""
    print("开始模型评估...")
    print("注意：这将加载已保存的模型并在环境中运行评估")
    
    # 创建评估器
    evaluator = ModelEvaluator(n_eval_episodes=100)
    
    # 评估所有模型
    results = evaluator.evaluate_all()
    
    print("\n评估完成！")
    
    return results


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()