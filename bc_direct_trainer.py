#!/usr/bin/env python3
"""
BC + Direct Training: 行为克隆预训练后直接在完整环境进行强化学习
避免复杂的课程学习阶段，直接在目标环境优化策略
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from collections import defaultdict, deque

# Add auction_sim to path
sys.path.append('.')

from auction_sim import config
from auction_sim.config import CurriculumStage, CURRICULUM_CONFIGS
from auction_sim.bc_trainer import BCTrainer
from auction_sim.bc_data_collector import BCDataCollector
from auction_sim.ma_runner import create_rule_agents, create_learning_agents
from auction_sim.ma_environment import MultiAgentAuctionEnv
from auction_sim.ma_trainer import MAPPOTrainer

class BCDirectTrainer:
    """BC预训练 + 直接在完整环境训练的混合trainer"""
    
    def __init__(self, 
                 use_bc_pretraining: bool = True,
                 bc_episodes: int = 10,
                 direct_episodes: int = 300,
                 target_stage: CurriculumStage = CurriculumStage.STAGE_8):
        """
        Args:
            use_bc_pretraining: 是否进行BC预训练
            bc_episodes: BC预训练的episodes数量
            direct_episodes: 直接训练的episodes数量  
            target_stage: 目标训练阶段（默认完整8智能体环境）
        """
        self.use_bc_pretraining = use_bc_pretraining
        self.bc_episodes = bc_episodes
        self.direct_episodes = direct_episodes
        self.target_stage = target_stage
        
        # 模型保存路径
        self.bc_model_path = "auction_sim/models/bc_pretrained.pth"
        self.models_dir = "auction_sim/models/bc_direct"
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"BC Direct Trainer初始化:")
        print(f"  BC预训练: {use_bc_pretraining} ({bc_episodes} episodes)")
        print(f"  直接训练: {direct_episodes} episodes")
        print(f"  目标环境: {target_stage.name}")

    def step1_bc_pretraining(self):
        """步骤1: 行为克隆预训练（如果需要）"""
        if not self.use_bc_pretraining:
            return None, None, None
            
        print("\n" + "="*70)
        print("步骤1: BC预训练")
        print("="*70)
        
        # 检查是否已存在BC模型
        if os.path.exists(self.bc_model_path):
            print(f"发现已存在的BC模型: {self.bc_model_path}")
            print("跳过BC预训练，直接使用现有模型")
            
            # 加载并验证模型
            try:
                torch.load(self.bc_model_path)
                print("✅ BC模型验证成功")
                return None, {'status': 'loaded_existing'}, {'accuracy': 'N/A', 'rmse': 'N/A'}
            except Exception as e:
                print(f"❌ BC模型加载失败: {e}")
                print("将重新进行BC预训练")
        
        # 进行BC预训练
        print("开始收集BC训练数据...")
        bc_collector = BCDataCollector()
        dataset = bc_collector.collect_dataset(
            n_episodes=self.bc_episodes,
            save_path="auction_sim/bc_dataset.pkl"
        )
        
        print("开始BC预训练...")
        bc_trainer = BCTrainer()
        bc_results = bc_trainer.train(
            dataset_path="auction_sim/bc_dataset.pkl",
            epochs=10,
            batch_size=64,
            learning_rate=1e-3,
            save_path=self.bc_model_path
        )
        
        # 提取关键指标
        bc_metrics = {
            'accuracy': bc_results.get('final_accuracy', 'N/A'),
            'rmse': bc_results.get('final_rmse', 'N/A'),
            'epochs': bc_results.get('epochs_trained', 'N/A')
        }
        
        print(f"\nBC预训练完成:")
        print(f"  最终准确率: {bc_metrics['accuracy']}")
        print(f"  最终RMSE: {bc_metrics['rmse']}")
        print(f"  训练epochs: {bc_metrics['epochs']}")
        
        return bc_trainer, bc_results, bc_metrics

    def step2_direct_training(self):
        """步骤2: 直接在目标环境进行强化学习"""
        print("\n" + "="*70)
        print("步骤2: 直接环境训练")
        print("="*70)
        
        # 获取目标阶段配置
        stage_config = CURRICULUM_CONFIGS[self.target_stage]
        print(f"目标环境: {stage_config['description']}")
        print(f"智能体配置: {stage_config['n_learning']}L + {stage_config['n_truthful']}T + "
              f"{stage_config['n_conservative']}C + {stage_config['n_aggressive']}A")
        
        # 创建智能体
        rule_agents = create_rule_agents(self.target_stage)
        learning_agents, learning_agent_ids = create_learning_agents(self.target_stage)
        
        print(f"创建了 {len(rule_agents)} 个规则智能体")
        print(f"创建了 {len(learning_agents)} 个学习智能体: {learning_agent_ids}")
        
        # 创建环境
        env = MultiAgentAuctionEnv(
            learning_agent_ids,
            rule_agents,
            curriculum_stage=self.target_stage
        )
        
        # 创建trainer - 使用适合直接训练的参数
        trainer = MAPPOTrainer(
            obs_dim=7,
            action_dim=1,
            n_agents=len(learning_agents),
            lr=3e-4,  # 略高的学习率，因为有BC基础
            gamma=0.95,
            gae_lambda=0.9,
            clip_ratio=0.2,  # 略大的clip ratio允许更多探索
            vf_coef=0.5,
            ent_coef=0.01,  # 适度的探索
            max_grad_norm=0.5,
            use_curriculum=False  # 不使用课程学习
        )
        
        # 如果使用BC预训练，加载BC模型权重
        if self.use_bc_pretraining and os.path.exists(self.bc_model_path):
            print("加载BC预训练权重...")
            bc_state_dict = torch.load(self.bc_model_path)
            
            # 将BC权重加载到每个学习智能体的Actor网络
            for agent_id in learning_agent_ids:
                if agent_id in trainer.networks:
                    # 只加载Actor部分的权重
                    actor_state_dict = {}
                    for key, value in bc_state_dict.items():
                        if 'actor' in key:
                            actor_state_dict[key] = value
                    
                    # 加载匹配的权重
                    trainer.networks[agent_id].load_state_dict(actor_state_dict, strict=False)
            
            print("✅ BC权重加载完成")
        
        # 连接模型到智能体
        for i, agent in enumerate(learning_agents):
            agent_id = f"Learning_{i}"
            
            class ModelWrapper:
                def __init__(self, network, trainer, agent_id):
                    self.network = network
                    self.trainer = trainer
                    self.agent_id = agent_id
                
                def predict(self, obs, deterministic=False):
                    action, _ = self.trainer.get_action(self.agent_id, obs, deterministic)
                    return np.array([action]), None
            
            model_wrapper = ModelWrapper(trainer.networks[agent_id], trainer, agent_id)
            agent.set_model(model_wrapper)
        
        # 训练循环
        print(f"\n开始直接训练 {self.direct_episodes} episodes...")
        
        # 性能跟踪
        episode_rewards = []
        episode_wins = []
        episode_metrics = []
        best_avg_reward = float('-inf')
        
        # 滑动窗口用于性能评估
        recent_rewards = deque(maxlen=20)
        recent_wins = deque(maxlen=20)
        recent_metrics = deque(maxlen=20)
        
        for episode in tqdm(range(self.direct_episodes), desc="Direct Training"):
            # 训练一个episode
            max_steps = stage_config['max_rounds']
            ep_rewards, ep_wins = trainer.train_episode(env, max_steps=max_steps)
            
            # 记录基本指标
            total_reward = sum(ep_rewards.values())
            total_wins = sum(ep_wins.values())
            avg_win_rate = total_wins / (len(learning_agents) * max_steps)
            
            episode_rewards.append(total_reward)
            episode_wins.append(avg_win_rate)
            recent_rewards.append(total_reward)
            recent_wins.append(avg_win_rate)
            
            # 计算详细指标
            episode_stats = self._calculate_episode_stats(env, learning_agent_ids, stage_config)
            episode_metrics.append(episode_stats)
            recent_metrics.append(episode_stats)
            
            # 保存最佳模型
            if total_reward > best_avg_reward:
                best_avg_reward = total_reward
                trainer.save_models(f"{self.models_dir}/best")
            
            # 定期保存checkpoint
            if (episode + 1) % 50 == 0:
                trainer.save_models(f"{self.models_dir}/checkpoint_ep{episode+1}")
            
            # 每20个episodes打印进度
            if (episode + 1) % 20 == 0 and len(recent_metrics) >= 10:
                self._print_progress(episode + 1, recent_rewards, recent_wins, recent_metrics)
        
        # 保存最终模型
        trainer.save_models(f"{self.models_dir}/final")
        
        # 绘制训练曲线
        self._plot_training_curves(episode_rewards, episode_wins, episode_metrics)
        
        print(f"\n直接训练完成！")
        print(f"最佳平均奖励: {best_avg_reward:.0f}")
        
        return trainer, learning_agents, rule_agents, episode_metrics

    def _calculate_episode_stats(self, env, learning_agent_ids, stage_config):
        """计算episode的详细统计数据"""
        stats = {
            'avg_win_rate': 0.0,
            'avg_roi': 0.0, 
            'budget_usage_ratio': 0.0,
            'avg_bid_ratio': 1.0,
            'individual_stats': {}
        }
        
        for agent_id in learning_agent_ids:
            agent_stats = {
                'win_rate': 0.0,
                'roi': -100.0,
                'budget_usage': 0.0,
                'bid_ratio': 1.0
            }
            
            # 计算预算使用率
            initial_budget = stage_config['budget']
            remaining = env.agent_budgets.get(agent_id, initial_budget)
            used = initial_budget - remaining
            agent_stats['budget_usage'] = used / initial_budget
            
            # 计算ROI（简化版本）
            if hasattr(env, 'agent_histories') and agent_id in env.agent_histories:
                history = env.agent_histories[agent_id]
                if history:
                    total_profit = sum(h.get('profit', 0) for h in history)
                    total_cost = sum(h.get('cost', 0) for h in history)
                    wins = sum(1 for h in history if h.get('won', False))
                    
                    agent_stats['win_rate'] = wins / len(history) if history else 0.0
                    if total_cost > 0:
                        agent_stats['roi'] = (total_profit / total_cost) * 100
                    
                    # 计算出价比率
                    bids = [h.get('bid', 0) for h in history if h.get('perceived_value', 0) > 0]
                    values = [h.get('perceived_value', 1) for h in history if h.get('perceived_value', 0) > 0]
                    if bids and values:
                        agent_stats['bid_ratio'] = np.mean([b/v for b, v in zip(bids, values) if v > 0])
            
            stats['individual_stats'][agent_id] = agent_stats
            
            # 累加到平均值
            stats['avg_win_rate'] += agent_stats['win_rate']
            stats['avg_roi'] += agent_stats['roi']  
            stats['budget_usage_ratio'] += agent_stats['budget_usage']
            stats['avg_bid_ratio'] += agent_stats['bid_ratio']
        
        # 计算平均值
        n_agents = len(learning_agent_ids)
        if n_agents > 0:
            stats['avg_win_rate'] /= n_agents
            stats['avg_roi'] /= n_agents
            stats['budget_usage_ratio'] /= n_agents
            stats['avg_bid_ratio'] /= n_agents
            
        return stats

    def _print_progress(self, episode, recent_rewards, recent_wins, recent_metrics):
        """打印训练进度"""
        avg_reward = np.mean(recent_rewards)
        avg_win_rate = np.mean(recent_wins)
        
        avg_roi = np.mean([m['avg_roi'] for m in recent_metrics])
        avg_budget = np.mean([m['budget_usage_ratio'] for m in recent_metrics])
        avg_bid_ratio = np.mean([m['avg_bid_ratio'] for m in recent_metrics])
        
        print(f"\nEpisode {episode}/{self.direct_episodes} (最近20集平均):")
        print(f"  总奖励: {avg_reward:.0f}")
        print(f"  胜率: {avg_win_rate:.1%}")
        print(f"  ROI: {avg_roi:.1f}%") 
        print(f"  预算使用: {avg_budget:.1%}")
        print(f"  出价比率: {avg_bid_ratio:.2f}")
        
        # 打印个体差异
        if recent_metrics:
            last_stats = recent_metrics[-1]['individual_stats']
            print("  个体表现:")
            for agent_id, stats in last_stats.items():
                print(f"    {agent_id}: 胜率{stats['win_rate']:.1%}, ROI{stats['roi']:.0f}%, "
                      f"预算{stats['budget_usage']:.1%}")

    def _plot_training_curves(self, episode_rewards, episode_wins, episode_metrics):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 12))
        
        # 提取指标数据
        episodes = range(1, len(episode_rewards) + 1)
        roi_values = [m['avg_roi'] for m in episode_metrics]
        budget_usage = [m['budget_usage_ratio'] for m in episode_metrics]
        bid_ratios = [m['avg_bid_ratio'] for m in episode_metrics]
        
        # 滑动平均
        def moving_average(data, window=10):
            return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
        
        ma_rewards = moving_average(episode_rewards)
        ma_wins = moving_average(episode_wins)
        ma_roi = moving_average(roi_values)
        ma_budget = moving_average(budget_usage)
        
        # 绘制子图
        plt.subplot(2, 3, 1)
        plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
        plt.plot(episodes, ma_rewards, color='blue', linewidth=2, label='MA-10')
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(episodes, episode_wins, alpha=0.3, color='green', label='Raw')
        plt.plot(episodes, ma_wins, color='green', linewidth=2, label='MA-10')
        plt.title('Win Rate')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 3)
        plt.plot(episodes, roi_values, alpha=0.3, color='red', label='Raw')
        plt.plot(episodes, ma_roi, color='red', linewidth=2, label='MA-10')
        plt.title('ROI (%)')
        plt.xlabel('Episode')
        plt.ylabel('ROI (%)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 4)
        plt.plot(episodes, budget_usage, alpha=0.3, color='orange', label='Raw')
        plt.plot(episodes, ma_budget, color='orange', linewidth=2, label='MA-10')
        plt.title('Budget Usage')
        plt.xlabel('Episode')
        plt.ylabel('Budget Usage Ratio')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 5)
        plt.plot(episodes, bid_ratios, alpha=0.3, color='purple', label='Bid/Value Ratio')
        plt.title('Bidding Behavior')
        plt.xlabel('Episode')
        plt.ylabel('Bid/Value Ratio')
        plt.legend()
        plt.grid(True)
        
        # 综合经济价值
        plt.subplot(2, 3, 6)
        economic_values = []
        for i, (reward, win, roi) in enumerate(zip(episode_rewards, episode_wins, roi_values)):
            # 简化的经济价值计算
            economic_value = 0.5 * reward + 0.35 * win * 1000 + 0.15 * max(0, roi) * 10
            economic_values.append(economic_value)
        
        ma_economic = moving_average(economic_values)
        plt.plot(episodes, economic_values, alpha=0.3, color='black', label='Raw')
        plt.plot(episodes, ma_economic, color='black', linewidth=2, label='MA-10')
        plt.title('Economic Value (Composite)')
        plt.xlabel('Episode')
        plt.ylabel('Economic Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('auction_sim/results/bc_direct_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存: auction_sim/results/bc_direct_training_curves.png")

    def train_complete_pipeline(self):
        """执行完整的BC+直接训练流程"""
        print("="*80)
        print("BC + DIRECT TRAINING PIPELINE")
        print("="*80)
        
        # 步骤1: BC预训练
        bc_trainer, bc_results, bc_metrics = self.step1_bc_pretraining()
        
        # 步骤2: 直接训练
        rl_trainer, learning_agents, rule_agents, training_metrics = self.step2_direct_training()
        
        # 最终评估
        print("\n" + "="*70)
        print("最终评估")
        print("="*70)
        
        if training_metrics:
            final_metrics = training_metrics[-20:]  # 最后20个episodes
            final_avg_reward = np.mean([sum(ep_rewards.values()) if hasattr(ep_rewards, 'values') 
                                      else ep_rewards for ep_rewards in [4000]])  # placeholder
            final_avg_win_rate = np.mean([m['avg_win_rate'] for m in final_metrics])
            final_avg_roi = np.mean([m['avg_roi'] for m in final_metrics])
            final_budget_usage = np.mean([m['budget_usage_ratio'] for m in final_metrics])
            
            print(f"最终性能 (最后20 episodes平均):")
            print(f"  平均奖励: {final_avg_reward:.0f}")
            print(f"  胜率: {final_avg_win_rate:.1%}")
            print(f"  ROI: {final_avg_roi:.1f}%")
            print(f"  预算使用: {final_budget_usage:.1%}")
            
            # 与理论期望对比
            theoretical_win_rate = 2 / 8  # 2个学习智能体在8个总智能体中
            print(f"\n对比分析:")
            print(f"  理论胜率: {theoretical_win_rate:.1%}")
            print(f"  实际胜率: {final_avg_win_rate:.1%}")
            print(f"  胜率效率: {final_avg_win_rate/theoretical_win_rate:.1%}")
        
        print(f"\n模型保存位置:")
        print(f"  最佳模型: {self.models_dir}/best")
        print(f"  最终模型: {self.models_dir}/final")
        print(f"  训练曲线: auction_sim/results/bc_direct_training_curves.png")
        
        return {
            'bc_trainer': bc_trainer,
            'bc_results': bc_results, 
            'bc_metrics': bc_metrics,
            'rl_trainer': rl_trainer,
            'learning_agents': learning_agents,
            'rule_agents': rule_agents,
            'training_metrics': training_metrics
        }

def main():
    """主函数 - 执行BC+直接训练"""
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("开始BC + Direct Training实验...")
    
    # 创建trainer
    trainer = BCDirectTrainer(
        use_bc_pretraining=True,   # 使用BC预训练
        bc_episodes=10,            # BC数据收集episodes
        direct_episodes=300,       # 直接训练episodes  
        target_stage=CurriculumStage.STAGE_8  # 完整8智能体环境
    )
    
    # 执行完整训练流程
    results = trainer.train_complete_pipeline()
    
    print("\n" + "="*80)
    print("BC + DIRECT TRAINING 实验完成！")
    print("="*80)

if __name__ == "__main__":
    main()