# /auction_sim/compare_algorithms.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .training_monitor import TrainingMonitor

class AlgorithmComparator:
    """比较不同强化学习算法性能的工具类"""
    
    def __init__(self, log_dir=None):
        """初始化比较器
        
        Args:
            log_dir: 日志目录路径，默认为项目根目录下的logs文件夹
        """
        if log_dir is None:
            # 默认使用项目根目录下的logs文件夹
            self.log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        else:
            self.log_dir = log_dir
            
        self.monitors = {}
        self.metrics_data = {}
    
    def load_monitors(self):
        """加载目录中的所有监控器数据"""
        if not os.path.exists(self.log_dir):
            print(f"日志目录不存在: {self.log_dir}")
            return
            
        # 查找所有监控器目录
        for item in os.listdir(self.log_dir):
            item_path = os.path.join(self.log_dir, item)
            if os.path.isdir(item_path):
                # 尝试加载监控器数据
                json_path = os.path.join(item_path, 'metrics.json')
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            # 提取算法名称
                            if '_PPO' in item:
                                algorithm = 'PPO'
                            elif '_DDPG' in item:
                                algorithm = 'DDPG'
                            else:
                                algorithm = 'Unknown'
                            
                            # 存储数据
                            self.metrics_data[item] = {
                                'data': data,
                                'algorithm': algorithm
                            }
                            print(f"已加载 {item} 的监控器数据")
                    except Exception as e:
                        print(f"加载 {item} 的监控器数据失败: {e}")
    
    def compare_metrics(self, metric_name, save_path=None):
        """比较不同算法在特定指标上的性能
        
        Args:
            metric_name: 要比较的指标名称
            save_path: 图表保存路径，如果为None则显示图表
        """
        if not self.metrics_data:
            print("没有可用的监控器数据，请先调用load_monitors()")
            return
            
        # 准备数据
        data_frames = []
        for monitor_name, info in self.metrics_data.items():
            if metric_name in info['data']:
                df = pd.DataFrame({
                    'Round': range(1, len(info['data'][metric_name]) + 1),
                    metric_name: info['data'][metric_name],
                    'Algorithm': info['algorithm'],
                    'Monitor': monitor_name
                })
                data_frames.append(df)
        
        if not data_frames:
            print(f"没有找到指标 {metric_name} 的数据")
            return
            
        # 合并数据
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        # 绘制图表
        plt.figure(figsize=(12, 6))
        sns.set_style('whitegrid')
        
        # 使用seaborn的lineplot，按算法分组
        ax = sns.lineplot(
            data=combined_df,
            x='Round',
            y=metric_name,
            hue='Algorithm',
            style='Algorithm',
            markers=True,
            dashes=False,
            ci='sd'  # 显示标准差
        )
        
        # 设置图表标题和标签
        plt.title(f'比较不同算法的{metric_name}性能', fontsize=16)
        plt.xlabel('回合数', fontsize=14)
        plt.ylabel(metric_name, fontsize=14)
        plt.legend(title='算法', fontsize=12)
        
        # 保存或显示图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到 {save_path}")
        else:
            plt.show()
    
    def compare_all_metrics(self, save_dir=None):
        """比较所有可用指标
        
        Args:
            save_dir: 图表保存目录，如果为None则显示图表
        """
        if not self.metrics_data:
            print("没有可用的监控器数据，请先调用load_monitors()")
            return
            
        # 获取所有可用指标
        all_metrics = set()
        for info in self.metrics_data.values():
            all_metrics.update(info['data'].keys())
        
        # 为每个指标创建比较图表
        for metric in all_metrics:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{metric}_comparison.png")
                self.compare_metrics(metric, save_path)
            else:
                self.compare_metrics(metric)


def main():
    """主函数，用于演示如何使用AlgorithmComparator"""
    # 创建比较器
    comparator = AlgorithmComparator()
    
    # 加载监控器数据
    comparator.load_monitors()
    
    # 创建保存目录
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'comparison_results')
    os.makedirs(save_dir, exist_ok=True)
    
    # 比较所有指标
    comparator.compare_all_metrics(save_dir)
    
    print(f"所有比较结果已保存到 {save_dir}")


if __name__ == '__main__':
    main()