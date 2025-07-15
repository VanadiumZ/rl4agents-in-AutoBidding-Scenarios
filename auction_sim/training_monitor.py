import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os
import json
from datetime import datetime


class TrainingMonitor:
    """
    训练监控器，用于记录和可视化强化学习智能体的训练过程
    """
    def __init__(self, agent_id: str, algorithm: str, log_dir: str, model_dir: Optional[str] = None):
        """
        初始化训练监控器
        
        Args:
            agent_id: 智能体的ID
            algorithm: 智能体使用的RL算法名称 (e.g., 'PPO', 'DDPG')
            log_dir: 日志和图表的输出目录
            model_dir: 模型保存目录 (可选，为了兼容runner.py的调用)
        """
        self.agent_id = agent_id
        self.algorithm = algorithm
        self.output_dir = log_dir # 将log_dir直接作为输出目录
        self.model_dir = model_dir # 存储它，即使监控器的核心功能不直接使用它
        
        self.metrics = {
            'round': [],
            'reward': [],
            'profit': [],
            'win_rate': [],
            'bid_value_ratio': [],
            'budget_remaining_ratio': []
        }
        self.episode_count = 0
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"TrainingMonitor for {agent_id} ({algorithm}) initialized. Output directory: {self.output_dir}")
        if self.model_dir:
            print(f"Associated Model directory: {self.model_dir}")
    
    def log_metrics(self, round_num: int, metrics: Dict):
        """
        记录训练指标
        
        Args:
            round_num: 当前轮次
            metrics: 指标字典，可包含 reward, profit, win_rate, bid_value_ratio, budget_remaining_ratio
        """
        self.metrics['round'].append(round_num)
        
        for metric_name in ['reward', 'profit', 'win_rate', 'bid_value_ratio', 'budget_remaining_ratio']:
            if metric_name in metrics:
                self.metrics[metric_name].append(metrics[metric_name])
            else:
                # 如果没有提供该指标，则填充None
                self.metrics[metric_name].append(None)
    
    def start_episode(self):
        """
        开始新的训练回合
        """
        self.episode_count += 1
        print(f"\nStarting episode {self.episode_count} for {self.agent_id}")
    
    def end_episode(self):
        """
        结束当前训练回合并保存结果
        """
        print(f"Episode {self.episode_count} for {self.agent_id} completed")
        self.save_metrics()
    
    def save_metrics(self):
        """
        保存训练指标
        """
        # 转换为DataFrame
        metrics_df = pd.DataFrame(self.metrics)
        
        # 保存为CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"metrics_ep{self.episode_count}_{timestamp}.csv")
        metrics_df.to_csv(csv_path, index=False)
        
        # 保存为JSON
        json_path = os.path.join(self.output_dir, f"metrics_ep{self.episode_count}_{timestamp}.json")
        with open(json_path, 'w') as f:
            # 将DataFrame转换为字典列表，以便JSON序列化
            json.dump(metrics_df.to_dict(orient='list'), f, indent=2)
        
        print(f"Metrics saved to {csv_path} and {json_path} for {self.agent_id}")
    
    def plot_training_curves(self, metrics_to_plot: Optional[List[str]] = None, 
                            smoothing: int = 10, output_file: Optional[str] = None):
        """
        绘制训练曲线
        
        Args:
            metrics_to_plot: 要绘制的指标列表，如果为None则绘制所有指标
            smoothing: 平滑窗口大小
            output_file: 输出文件路径，如果为None则显示图表
        """
        if not self.metrics['round']:
            print(f"No metrics available for {self.agent_id}. Skipping plot.")
            return
        
        # 如果未指定要绘制的指标，则绘制所有非None的指标
        if metrics_to_plot is None:
            metrics_to_plot = []
            for metric_name in ['reward', 'profit', 'win_rate', 'bid_value_ratio', 'budget_remaining_ratio']:
                if any(x is not None for x in self.metrics[metric_name]):
                    metrics_to_plot.append(metric_name)
        
        if not metrics_to_plot:
            print(f"No valid metrics to plot for {self.agent_id}. Skipping plot.")
            return

        # 创建图表
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 4 * len(metrics_to_plot)), sharex=True)
        if len(metrics_to_plot) == 1:
            axes = [axes]  # 确保axes始终是列表
        
        # 绘制每个指标
        for i, metric_name in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # 获取指标数据
            rounds = np.array(self.metrics['round'])
            metric_values = np.array(self.metrics[metric_name], dtype=float)
            
            # 过滤掉None值
            valid_indices = ~np.isnan(metric_values)
            valid_rounds = rounds[valid_indices]
            valid_values = metric_values[valid_indices]
            
            if len(valid_values) > 0:
                # 绘制原始数据
                ax.plot(valid_rounds, valid_values, 'o-', alpha=0.3, label='Raw')
                
                # 如果有足够的数据点，绘制平滑曲线
                if len(valid_values) > smoothing:
                    # 计算移动平均
                    smooth_values = np.convolve(valid_values, np.ones(smoothing)/smoothing, mode='valid')
                    smooth_rounds = valid_rounds[smoothing-1:]
                    
                    # 绘制平滑曲线
                    ax.plot(smooth_rounds, smooth_values, '-', linewidth=2, 
                           label=f'Moving Avg (window={smoothing})')
                else:
                    ax.plot(valid_rounds, valid_values, '-', linewidth=2, label='Raw') # 如果数据不够平滑，也画线
                
                # 设置标题和标签
                ax.set_title(f'{metric_name.replace("_", " ").title()} for {self.agent_id}')
                ax.set_ylabel(metric_name)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
            else:
                ax.text(0.5, 0.5, f'No valid {metric_name} data', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
        
        # 设置x轴标签
        axes[-1].set_xlabel('Round')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图表
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {output_file} for {self.agent_id}")
        else:
            plt.show()
    
    def plot_comparison(self, other_monitors: List['TrainingMonitor'], 
                       labels: List[str], metric: str = 'reward',
                       smoothing: int = 10, output_file: Optional[str] = None):
        """
        比较多个训练监控器的指标
        
        Args:
            other_monitors: 其他训练监控器列表
            labels: 每个监控器的标签
            metric: 要比较的指标
            smoothing: 平滑窗口大小
            output_file: 输出文件路径，如果为None则显示图表
        """
        if len(other_monitors) + 1 != len(labels):
            raise ValueError("Number of monitors plus this monitor must match number of labels")
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 绘制当前监控器的指标
        self._plot_metric_for_comparison(self.metrics, labels[0], metric, smoothing)
        
        # 绘制其他监控器的指标
        for i, monitor in enumerate(other_monitors):
            self._plot_metric_for_comparison(monitor.metrics, labels[i+1], metric, smoothing)
        
        # 设置标题和标签
        plt.title(f'Comparison of {metric.replace("_", " ").title()}')
        plt.xlabel('Round')
        plt.ylabel(metric)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存或显示图表
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {output_file}")
        else:
            plt.show()
    
    def _plot_metric_for_comparison(self, metrics: Dict, label: str, 
                                  metric: str, smoothing: int):
        """
        为比较绘制单个指标
        
        Args:
            metrics: 指标字典
            label: 标签
            metric: 要绘制的指标
            smoothing: 平滑窗口大小
        """
        # 获取指标数据
        rounds = np.array(metrics['round'])
        metric_values = np.array(metrics[metric], dtype=float)
        
        # 过滤掉None值
        valid_indices = ~np.isnan(metric_values)
        valid_rounds = rounds[valid_indices]
        valid_values = metric_values[valid_indices]
        
        if len(valid_values) > 0:
            # 绘制原始数据
            plt.plot(valid_rounds, valid_values, 'o-', alpha=0.2)
            
            # 如果有足够的数据点，绘制平滑曲线
            if len(valid_values) > smoothing:
                # 计算移动平均
                smooth_values = np.convolve(valid_values, np.ones(smoothing)/smoothing, mode='valid')
                smooth_rounds = valid_rounds[smoothing-1:]
                
                # 绘制平滑曲线
                plt.plot(smooth_rounds, smooth_values, '-', linewidth=2, label=label)
            else:
                plt.plot(valid_rounds, valid_values, '-', linewidth=2, label=label)


def main():
    """
    训练监控器示例
    """
    # 创建训练监控器
    monitor = TrainingMonitor(agent_id="TestAgent", algorithm="TestAlgo", log_dir="training_logs/test_agent")
    
    # 模拟训练过程
    monitor.start_episode()
    
    for round_num in range(1, 101):
        # 模拟一些指标
        reward = np.sin(round_num / 10) + np.random.normal(0, 0.2)
        profit = round_num * 0.1 + np.random.normal(0, 1)
        win_rate = 0.5 + 0.3 * np.sin(round_num / 20) + np.random.normal(0, 0.05)
        
        # 记录指标
        monitor.log_metrics(round_num, {
            'reward': reward,
            'profit': profit,
            'win_rate': win_rate
        })
    
    monitor.end_episode()
    
    # 绘制训练曲线
    monitor.plot_training_curves(output_file="training_logs/test_agent/training_curves.png")
    
    print("\nTraining monitoring example completed!")


if __name__ == "__main__":
    main()