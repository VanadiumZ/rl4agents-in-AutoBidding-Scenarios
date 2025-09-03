#!/usr/bin/env python3
"""
BC + IPPO Training: 行为克隆预训练后使用独立PPO训练
每个智能体独立训练，不共享参数，解决不对称问题
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
from auction_sim.ippo_trainer import IPPOTrainer, IPPOActorCriticNetwork
from auction_sim.ma_environment import MultiAgentAuctionEnv
from auction_sim.agents import TruthfulAgent, ConservativeAgent, AggressiveAgent, MultiAgentLearningAgent

class BCIPPOTrainer:
    """BC预训练 + 独立PPO训练的混合trainer"""
    
    def __init__(self,
                 use_bc_pretraining: bool = True,
                 bc_episodes: int = 10,
                 ippo_episodes: int = 300,
                 target_stage: CurriculumStage = CurriculumStage.STAGE_8):
        """
        Args:
            use_bc_pretraining: 是否使用BC预训练
            bc_episodes: BC数据收集episodes
            ippo_episodes: IPPO训练episodes  
            target_stage: 目标环境阶段
        """
        self.use_bc_pretraining = use_bc_pretraining
        self.bc_episodes = bc_episodes
        self.ippo_episodes = ippo_episodes
        self.target_stage = target_stage
        
        # 模型路径
        self.bc_model_path = "auction_sim/models/bc_pretrained.pth"
        self.models_dir = "auction_sim/models/bc_ippo"
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"BC + IPPO Trainer初始化:")
        print(f"  BC预训练: {use_bc_pretraining}")
        print(f"  IPPO训练: {ippo_episodes} episodes")
        print(f"  目标环境: {target_stage.name}")
        print(f"  关键特性: 独立网络，不共享参数")

    def step1_bc_pretraining(self):
        """步骤1: BC预训练（如果需要）"""
        if not self.use_bc_pretraining:
            return None, None, None
            
        print("\n" + "="*70)
        print("步骤1: BC预训练")
        print("="*70)
        
        # 检查现有BC模型
        if os.path.exists(self.bc_model_path):
            print(f"发现已存在的BC模型: {self.bc_model_path}")
            print("跳过BC预训练，直接使用现有模型")
            
            try:
                bc_state_dict = torch.load(self.bc_model_path)
                print(f"✅ BC模型加载成功，包含 {len(bc_state_dict)} 个参数")
                return None, {'status': 'loaded_existing'}, {'model_path': self.bc_model_path}
            except Exception as e:
                print(f"❌ BC模型加载失败: {e}")
                print("将重新进行BC预训练")
        
        # 收集BC数据
        print("开始收集BC训练数据...")
        bc_collector = BCDataCollector()
        dataset = bc_collector.collect_dataset(
            n_episodes=self.bc_episodes,
            save_path="auction_sim/bc_dataset.pkl"
        )
        
        # BC训练
        print("开始BC预训练...")
        bc_trainer = BCTrainer()
        bc_results = bc_trainer.train(
            dataset_path="auction_sim/bc_dataset.pkl",
            epochs=10,
            batch_size=64,
            learning_rate=1e-3,
            save_path=self.bc_model_path
        )
        
        bc_metrics = {
            'accuracy': bc_results.get('final_accuracy', 'N/A'),
            'rmse': bc_results.get('final_rmse', 'N/A'),
            'model_path': self.bc_model_path
        }
        
        print(f"\nBC预训练完成:")
        print(f"  准确率: {bc_metrics['accuracy']}")
        print(f"  RMSE: {bc_metrics['rmse']}")
        
        return bc_trainer, bc_results, bc_metrics

    def step2_ippo_training(self):
        """步骤2: 独立PPO训练"""
        print("\n" + "="*70)
        print("步骤2: 独立PPO训练 (IPPO)")
        print("="*70)
        
        # 获取环境配置
        stage_config = CURRICULUM_CONFIGS[self.target_stage]
        print(f"目标环境: {stage_config['description']}")
        print(f"智能体配置: {stage_config['n_learning']}L + {stage_config['n_truthful']}T + "
              f"{stage_config['n_conservative']}C + {stage_config['n_aggressive']}A")
        
        # 创建规则智能体
        rule_agents = self._create_rule_agents(stage_config)
        
        # 创建学习智能体
        learning_agents = []
        learning_agent_ids = []
        for i in range(stage_config['n_learning']):
            agent_id = f"Learning_{i}"
            agent = MultiAgentLearningAgent(
                agent_id=agent_id,
                budget=stage_config['budget'],
                perception_noise_std=config.AGENT_PERCEPTION_NOISE_STD,
                model=None,
                is_training=True
            )
            learning_agents.append(agent)
            learning_agent_ids.append(agent_id)
        
        print(f"创建了 {len(rule_agents)} 个规则智能体")
        print(f"创建了 {len(learning_agents)} 个学习智能体: {learning_agent_ids}")
        
        # 创建环境
        env = MultiAgentAuctionEnv(
            learning_agent_ids,
            rule_agents,
            curriculum_stage=self.target_stage
        )
        
        # 创建IPPO trainer - 关键：每个智能体独立的网络
        trainer = IPPOTrainer(
            obs_dim=7,
            action_dim=1,
            n_agents=len(learning_agents),
            lr=3e-4,  # 与MAPPO一致
            gamma=0.95,
            gae_lambda=0.9,
            clip_ratio=0.2,
            vf_coef=0.5,
            ent_coef=0.01,  # 适度探索
            max_grad_norm=0.5
        )
        
        print("\n✨ IPPO关键特性:")
        print("  - 每个智能体完全独立的Actor-Critic网络")
        print("  - 独立的优化器和经验缓冲区")
        print("  - 无参数共享，避免梯度冲突")
        
        # 如果使用BC预训练，加载权重到每个独立网络
        if self.use_bc_pretraining and os.path.exists(self.bc_model_path):
            print("\n加载BC预训练权重到每个独立网络...")
            bc_state_dict = torch.load(self.bc_model_path)
            
            for agent_id in learning_agent_ids:
                if agent_id in trainer.networks:
                    # 为每个智能体的独立网络加载BC权重
                    # 注意：BC模型和IPPO网络结构可能不完全匹配
                    try:
                        # 提取actor相关权重
                        actor_weights = {}
                        for key, value in bc_state_dict.items():
                            # 映射BC权重到IPPO网络结构
                            if 'actor' in key:
                                # 调整键名以匹配IPPO网络
                                new_key = key.replace('actor.', '')
                                if 'linear' in new_key:
                                    new_key = 'actor_linear' + new_key.split('linear')[-1]
                                actor_weights[new_key] = value
                            elif 'backbone' in key or 'feature' in key:
                                actor_weights[key] = value
                        
                        # 加载权重（允许部分匹配）
                        trainer.networks[agent_id].load_state_dict(actor_weights, strict=False)
                        print(f"  ✅ {agent_id}: BC权重加载成功")
                    except Exception as e:
                        print(f"  ⚠️ {agent_id}: BC权重部分加载 ({e})")
                        # 即使部分失败也继续，IPPO会从这个基础上训练
            
            print("BC权重初始化完成，每个智能体从相同起点开始独立演化")
        
        # 连接模型到智能体
        for i, agent in enumerate(learning_agents):
            agent_id = f"Learning_{i}"
            
            class IPPOModelWrapper:
                def __init__(self, network, trainer, agent_id):
                    self.network = network
                    self.trainer = trainer
                    self.agent_id = agent_id
                
                def predict(self, obs, deterministic=False):
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        action, _ = self.network.get_action(obs_tensor, deterministic)
                    return action.cpu().numpy(), None
            
            model_wrapper = IPPOModelWrapper(trainer.networks[agent_id], trainer, agent_id)
            agent.set_model(model_wrapper)
        
        # 训练循环
        print(f"\n开始IPPO独立训练 {self.ippo_episodes} episodes...")
        
        # 性能跟踪
        episode_rewards = []
        episode_wins = []
        episode_metrics = []
        best_avg_reward = float('-inf')
        
        # 滑动窗口
        recent_rewards = deque(maxlen=20)
        recent_wins = deque(maxlen=20)
        recent_metrics = deque(maxlen=20)
        
        # 个体性能跟踪 - 关键：分别跟踪每个智能体
        individual_performance = {
            agent_id: {
                'rewards': [],
                'win_rates': [],
                'rois': [],
                'budget_usage': []
            } for agent_id in learning_agent_ids
        }
        
        pbar = tqdm(range(self.ippo_episodes), desc="IPPO Training")
        
        for episode in pbar:
            # 训练一个episode
            max_steps = stage_config['max_rounds']
            ep_rewards, ep_wins = trainer.train_episode(env, max_steps=max_steps)
            
            # 计算统计
            total_reward = sum(ep_rewards.values())
            total_wins = sum(ep_wins.values())
            avg_win_rate = total_wins / (len(learning_agents) * max_steps)
            
            episode_rewards.append(total_reward)
            episode_wins.append(avg_win_rate)
            recent_rewards.append(total_reward)
            recent_wins.append(avg_win_rate)
            
            # 计算详细指标（包括个体指标）
            episode_stats = self._calculate_episode_stats(env, learning_agent_ids, stage_config, ep_rewards, ep_wins)
            episode_metrics.append(episode_stats)
            recent_metrics.append(episode_stats)
            
            # 记录个体性能
            for agent_id in learning_agent_ids:
                if agent_id in episode_stats['individual_stats']:
                    stats = episode_stats['individual_stats'][agent_id]
                    individual_performance[agent_id]['rewards'].append(ep_rewards.get(agent_id, 0))
                    individual_performance[agent_id]['win_rates'].append(stats['win_rate'])
                    individual_performance[agent_id]['rois'].append(stats['roi'])
                    individual_performance[agent_id]['budget_usage'].append(stats['budget_usage'])
            
            # 更新进度条
            if len(recent_metrics) > 0:
                last_metrics = recent_metrics[-1]
                pbar.set_postfix({
                    'Reward': f"{total_reward:.0f}",
                    'WinRate': f"{avg_win_rate:.1%}",
                    'L0_WR': f"{last_metrics['individual_stats'].get('Learning_0', {}).get('win_rate', 0):.1%}",
                    'L1_WR': f"{last_metrics['individual_stats'].get('Learning_1', {}).get('win_rate', 0):.1%}"
                })
            
            # 保存最佳模型
            if total_reward > best_avg_reward:
                best_avg_reward = total_reward
                trainer.save_models(f"{self.models_dir}/best")
            
            # 定期checkpoint
            if (episode + 1) % 50 == 0:
                trainer.save_models(f"{self.models_dir}/checkpoint_ep{episode+1}")
            
            # 定期打印详细进度
            if (episode + 1) % 20 == 0 and len(recent_metrics) >= 10:
                self._print_progress(episode + 1, recent_rewards, recent_wins, recent_metrics, individual_performance)
        
        pbar.close()
        
        # 保存最终模型
        trainer.save_models(f"{self.models_dir}/final")
        
        # 绘制训练曲线
        self._plot_training_curves(episode_rewards, episode_wins, episode_metrics, individual_performance)
        
        print(f"\nIPPO训练完成！")
        print(f"最佳平均奖励: {best_avg_reward:.0f}")
        
        # 分析不对称性
        self._analyze_asymmetry(individual_performance)
        
        return trainer, learning_agents, rule_agents, episode_metrics, individual_performance

    def _create_rule_agents(self, stage_config):
        """创建规则智能体"""
        rule_agents = []
        agent_id_counter = 0
        
        # Truthful agents
        for _ in range(stage_config.get('n_truthful', 0)):
            agent = TruthfulAgent(
                f"Truthful_{agent_id_counter}",
                stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD
            )
            rule_agents.append(agent)
            agent_id_counter += 1
        
        # Conservative agents
        for _ in range(stage_config.get('n_conservative', 0)):
            agent = ConservativeAgent(
                f"Conservative_{agent_id_counter}",
                stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD,
                stage_config['max_rounds']
            )
            rule_agents.append(agent)
            agent_id_counter += 1
        
        # Aggressive agents
        for _ in range(stage_config.get('n_aggressive', 0)):
            total_agents = (stage_config.get('n_learning', 0) + 
                          stage_config.get('n_truthful', 0) + 
                          stage_config.get('n_conservative', 0) + 
                          stage_config.get('n_aggressive', 0))
            agent = AggressiveAgent(
                f"Aggressive_{agent_id_counter}",
                stage_config['budget'],
                config.AGENT_PERCEPTION_NOISE_STD,
                total_agents
            )
            rule_agents.append(agent)
            agent_id_counter += 1
        
        return rule_agents

    def _calculate_episode_stats(self, env, learning_agent_ids, stage_config, ep_rewards, ep_wins):
        """计算episode统计数据"""
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
            
            # 预算使用率
            initial_budget = stage_config['budget']
            remaining = env.agent_budgets.get(agent_id, initial_budget)
            used = initial_budget - remaining
            agent_stats['budget_usage'] = used / initial_budget
            
            # 胜率和ROI
            if hasattr(env, 'agent_histories') and agent_id in env.agent_histories:
                history = env.agent_histories[agent_id]
                if history:
                    total_profit = sum(h.get('profit', 0) for h in history)
                    total_cost = sum(h.get('cost', 0) for h in history)
                    wins = sum(1 for h in history if h.get('won', False))
                    
                    agent_stats['win_rate'] = wins / len(history) if history else 0.0
                    if total_cost > 0:
                        agent_stats['roi'] = (total_profit / total_cost) * 100
                    
                    # 出价比率
                    bids = [h.get('bid', 0) for h in history if h.get('perceived_value', 0) > 0]
                    values = [h.get('perceived_value', 1) for h in history if h.get('perceived_value', 0) > 0]
                    if bids and values:
                        agent_stats['bid_ratio'] = np.mean([b/v for b, v in zip(bids, values) if v > 0])
            
            stats['individual_stats'][agent_id] = agent_stats
            
            # 累加平均值
            stats['avg_win_rate'] += agent_stats['win_rate']
            stats['avg_roi'] += agent_stats['roi']
            stats['budget_usage_ratio'] += agent_stats['budget_usage']
            stats['avg_bid_ratio'] += agent_stats['bid_ratio']
        
        # 计算平均
        n_agents = len(learning_agent_ids)
        if n_agents > 0:
            stats['avg_win_rate'] /= n_agents
            stats['avg_roi'] /= n_agents
            stats['budget_usage_ratio'] /= n_agents
            stats['avg_bid_ratio'] /= n_agents
        
        return stats

    def _print_progress(self, episode, recent_rewards, recent_wins, recent_metrics, individual_performance):
        """打印训练进度"""
        avg_reward = np.mean(recent_rewards)
        avg_win_rate = np.mean(recent_wins)
        
        avg_roi = np.mean([m['avg_roi'] for m in recent_metrics])
        avg_budget = np.mean([m['budget_usage_ratio'] for m in recent_metrics])
        
        print(f"\nEpisode {episode}/{self.ippo_episodes} (最近20集平均):")
        print(f"  总奖励: {avg_reward:.0f}")
        print(f"  平均胜率: {avg_win_rate:.1%}")
        print(f"  平均ROI: {avg_roi:.1f}%")
        print(f"  预算使用: {avg_budget:.1%}")
        
        # 关键：显示个体表现对比
        print("\n  🔍 个体表现对比 (IPPO独立训练):")
        for agent_id in individual_performance.keys():
            recent_wrs = individual_performance[agent_id]['win_rates'][-20:]
            recent_rois = individual_performance[agent_id]['rois'][-20:]
            recent_budgets = individual_performance[agent_id]['budget_usage'][-20:]
            
            if recent_wrs:
                print(f"    {agent_id}:")
                print(f"      胜率: {np.mean(recent_wrs):.1%}")
                print(f"      ROI: {np.mean(recent_rois):.0f}%")
                print(f"      预算: {np.mean(recent_budgets):.1%}")
        
        # 计算不对称度
        if len(individual_performance) == 2:
            agents = list(individual_performance.keys())
            wr_diff = abs(np.mean(individual_performance[agents[0]]['win_rates'][-20:]) - 
                         np.mean(individual_performance[agents[1]]['win_rates'][-20:]))
            print(f"\n  📊 不对称度: 胜率差异 {wr_diff:.1%}")

    def _analyze_asymmetry(self, individual_performance):
        """分析智能体不对称性"""
        print("\n" + "="*70)
        print("不对称性分析 (IPPO vs MAPPO对比)")
        print("="*70)
        
        if len(individual_performance) == 2:
            agents = list(individual_performance.keys())
            
            # 最终性能
            final_episodes = 50  # 最后50个episodes
            
            agent0_final_wr = np.mean(individual_performance[agents[0]]['win_rates'][-final_episodes:])
            agent1_final_wr = np.mean(individual_performance[agents[1]]['win_rates'][-final_episodes:])
            
            agent0_final_roi = np.mean(individual_performance[agents[0]]['rois'][-final_episodes:])
            agent1_final_roi = np.mean(individual_performance[agents[1]]['rois'][-final_episodes:])
            
            print(f"\n最终性能对比 (最后{final_episodes}集):")
            print(f"  {agents[0]}: 胜率 {agent0_final_wr:.1%}, ROI {agent0_final_roi:.0f}%")
            print(f"  {agents[1]}: 胜率 {agent1_final_wr:.1%}, ROI {agent1_final_roi:.0f}%")
            
            wr_diff = abs(agent0_final_wr - agent1_final_wr)
            roi_diff = abs(agent0_final_roi - agent1_final_roi)
            
            print(f"\n不对称度指标:")
            print(f"  胜率差异: {wr_diff:.1%}")
            print(f"  ROI差异: {roi_diff:.0f}%")
            
            # 评估
            if wr_diff < 0.05:  # 5%以内
                print("\n✅ 评估: 不对称性问题基本解决！")
            elif wr_diff < 0.15:  # 15%以内
                print("\n⚠️ 评估: 轻度不对称，可接受")
            else:
                print("\n❌ 评估: 仍存在显著不对称")
            
            # 对比MAPPO结果
            print("\n与MAPPO对比:")
            print("  MAPPO: 胜率差异通常 30-40%")
            print(f"  IPPO:  胜率差异 {wr_diff:.1%}")
            improvement = max(0, 0.35 - wr_diff) / 0.35 * 100
            print(f"  改善程度: {improvement:.0f}%")

    def _plot_training_curves(self, episode_rewards, episode_wins, episode_metrics, individual_performance):
        """绘制训练曲线"""
        plt.figure(figsize=(18, 12))
        
        episodes = range(1, len(episode_rewards) + 1)
        
        # 滑动平均
        def moving_average(data, window=10):
            return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
        
        # 1. 总奖励
        plt.subplot(2, 3, 1)
        plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
        plt.plot(episodes, moving_average(episode_rewards), color='blue', linewidth=2, label='MA-10')
        plt.title('Episode Rewards (IPPO)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 平均胜率
        plt.subplot(2, 3, 2)
        plt.plot(episodes, episode_wins, alpha=0.3, color='green', label='Average')
        plt.plot(episodes, moving_average(episode_wins), color='green', linewidth=2, label='MA-10')
        plt.title('Average Win Rate (IPPO)')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 个体胜率对比 - 关键图表
        plt.subplot(2, 3, 3)
        colors = ['red', 'blue']
        for i, (agent_id, perf) in enumerate(individual_performance.items()):
            plt.plot(episodes, perf['win_rates'], alpha=0.3, color=colors[i])
            plt.plot(episodes, moving_average(perf['win_rates']), 
                    color=colors[i], linewidth=2, label=agent_id)
        plt.title('Individual Win Rates (IPPO)')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 个体ROI对比
        plt.subplot(2, 3, 4)
        for i, (agent_id, perf) in enumerate(individual_performance.items()):
            plt.plot(episodes, perf['rois'], alpha=0.3, color=colors[i])
            plt.plot(episodes, moving_average(perf['rois']), 
                    color=colors[i], linewidth=2, label=agent_id)
        plt.title('Individual ROI (IPPO)')
        plt.xlabel('Episode')
        plt.ylabel('ROI (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 胜率差异度量
        plt.subplot(2, 3, 5)
        if len(individual_performance) == 2:
            agents = list(individual_performance.keys())
            wr_diffs = [abs(individual_performance[agents[0]]['win_rates'][i] - 
                           individual_performance[agents[1]]['win_rates'][i]) 
                       for i in range(len(episode_rewards))]
            plt.plot(episodes, wr_diffs, alpha=0.3, color='purple')
            plt.plot(episodes, moving_average(wr_diffs), color='purple', linewidth=2)
            plt.axhline(y=0.05, color='green', linestyle='--', label='目标 (<5%)')
            plt.axhline(y=0.15, color='orange', linestyle='--', label='可接受 (<15%)')
            plt.title('Win Rate Asymmetry (IPPO)')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate Difference')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 6. 预算使用对比
        plt.subplot(2, 3, 6)
        for i, (agent_id, perf) in enumerate(individual_performance.items()):
            plt.plot(episodes, perf['budget_usage'], alpha=0.3, color=colors[i])
            plt.plot(episodes, moving_average(perf['budget_usage']), 
                    color=colors[i], linewidth=2, label=agent_id)
        plt.title('Budget Usage (IPPO)')
        plt.xlabel('Episode')
        plt.ylabel('Budget Usage Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('auction_sim/results/bc_ippo_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存: auction_sim/results/bc_ippo_training_curves.png")

    def train_complete_pipeline(self):
        """执行完整的BC+IPPO训练流程"""
        print("="*80)
        print("BC + IPPO TRAINING PIPELINE")
        print("独立PPO解决不对称问题")
        print("="*80)
        
        # 步骤1: BC预训练
        bc_trainer, bc_results, bc_metrics = self.step1_bc_pretraining()
        
        # 步骤2: IPPO独立训练
        ippo_trainer, learning_agents, rule_agents, training_metrics, individual_performance = self.step2_ippo_training()
        
        # 最终评估
        print("\n" + "="*70)
        print("最终评估")
        print("="*70)
        
        if training_metrics:
            final_metrics = training_metrics[-20:]  # 最后20个episodes
            final_avg_win_rate = np.mean([m['avg_win_rate'] for m in final_metrics])
            final_avg_roi = np.mean([m['avg_roi'] for m in final_metrics])
            final_budget_usage = np.mean([m['budget_usage_ratio'] for m in final_metrics])
            
            print(f"最终性能 (最后20 episodes平均):")
            print(f"  平均胜率: {final_avg_win_rate:.1%}")
            print(f"  平均ROI: {final_avg_roi:.1f}%")
            print(f"  预算使用: {final_budget_usage:.1%}")
            
            # 个体最终性能
            print(f"\n个体最终性能:")
            for agent_id, perf in individual_performance.items():
                final_wr = np.mean(perf['win_rates'][-20:])
                final_roi = np.mean(perf['rois'][-20:])
                print(f"  {agent_id}: 胜率 {final_wr:.1%}, ROI {final_roi:.0f}%")
            
            # 与理论期望对比
            theoretical_win_rate = 2 / 8  # 2个学习智能体在8个总智能体中
            print(f"\n对比分析:")
            print(f"  理论胜率: {theoretical_win_rate:.1%}")
            print(f"  实际平均胜率: {final_avg_win_rate:.1%}")
            print(f"  胜率效率: {final_avg_win_rate/theoretical_win_rate:.1%}")
        
        print(f"\n模型保存位置:")
        print(f"  最佳模型: {self.models_dir}/best")
        print(f"  最终模型: {self.models_dir}/final")
        print(f"  训练曲线: auction_sim/results/bc_ippo_training_curves.png")
        
        return {
            'bc_trainer': bc_trainer,
            'bc_results': bc_results,
            'bc_metrics': bc_metrics,
            'ippo_trainer': ippo_trainer,
            'learning_agents': learning_agents,
            'rule_agents': rule_agents,
            'training_metrics': training_metrics,
            'individual_performance': individual_performance
        }

def main():
    """主函数 - 执行BC+IPPO训练"""
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("开始BC + IPPO Training实验...")
    print("目标: 验证独立训练能否解决智能体不对称问题")
    
    # 创建trainer
    trainer = BCIPPOTrainer(
        use_bc_pretraining=True,    # 使用BC预训练
        bc_episodes=10,             # BC数据收集（如果需要）
        ippo_episodes=300,          # IPPO训练episodes
        target_stage=CurriculumStage.STAGE_8  # 完整8智能体环境
    )
    
    # 执行训练
    results = trainer.train_complete_pipeline()
    
    print("\n" + "="*80)
    print("BC + IPPO 实验完成！")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()