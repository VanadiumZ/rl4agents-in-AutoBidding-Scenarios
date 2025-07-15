import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import json
from datetime import datetime

from auction_sim import config
from auction_sim.runner import main as run_simulation


class HyperparameterTuner:
    """
    超参数搜索类，用于寻找最优的强化学习参数配置
    """
    def __init__(self, param_grid: Dict, n_trials: int = 10, output_dir: str = "tuning_results"):
        """
        初始化超参数搜索器
        
        Args:
            param_grid: 参数网格，格式为 {param_name: [param_values]}
            n_trials: 每个参数组合的重复试验次数
            output_dir: 结果输出目录
        """
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.output_dir = output_dir
        self.results = []
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_param_combinations(self) -> List[Dict]:
        """
        生成所有参数组合
        
        Returns:
            参数组合列表
        """
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = []
        
        def _generate_combinations(current_combo, index):
            if index == len(param_names):
                combinations.append(current_combo.copy())
                return
            
            for value in param_values[index]:
                current_combo[param_names[index]] = value
                _generate_combinations(current_combo, index + 1)
        
        _generate_combinations({}, 0)
        return combinations
    
    def run_trial(self, params: Dict) -> Dict:
        """
        运行单次试验
        
        Args:
            params: 参数配置
            
        Returns:
            试验结果
        """
        # 备份原始配置
        original_config = {}
        for param_name in params:
            if hasattr(config, param_name):
                original_config[param_name] = getattr(config, param_name)
        
        # 设置新参数
        for param_name, param_value in params.items():
            setattr(config, param_name, param_value)
        
        # 运行模拟
        try:
            result = run_simulation(return_metrics=True)
            
            # 添加参数信息到结果中
            result.update(params)
            
            return result
        finally:
            # 恢复原始配置
            for param_name, param_value in original_config.items():
                setattr(config, param_name, param_value)
    
    def run(self) -> pd.DataFrame:
        """
        运行超参数搜索
        
        Returns:
            包含所有试验结果的DataFrame
        """
        param_combinations = self.generate_param_combinations()
        print(f"Generated {len(param_combinations)} parameter combinations")
        
        for i, params in enumerate(param_combinations):
            print(f"\nTesting combination {i+1}/{len(param_combinations)}: {params}")
            
            for trial in range(self.n_trials):
                print(f"  Trial {trial+1}/{self.n_trials}")
                result = self.run_trial(params)
                result['trial'] = trial
                self.results.append(result)
                
                # 保存中间结果
                self._save_results()
        
        # 转换结果为DataFrame
        results_df = pd.DataFrame(self.results)
        
        # 保存最终结果
        self._save_results(results_df)
        
        return results_df
    
    def _save_results(self, results_df: pd.DataFrame = None):
        """
        保存结果
        
        Args:
            results_df: 结果DataFrame，如果为None则从self.results创建
        """
        if results_df is None:
            results_df = pd.DataFrame(self.results)
        
        # 保存为CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"tuning_results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        
        # 保存为JSON
        json_path = os.path.join(self.output_dir, f"tuning_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def analyze_results(self, metric: str = 'learning_agent_profit') -> Tuple[Dict, pd.DataFrame]:
        """
        分析结果，找出最优参数组合
        
        Args:
            metric: 用于评估的指标
            
        Returns:
            (最优参数组合, 汇总结果DataFrame)
        """
        if not self.results:
            raise ValueError("No results available. Run the tuner first.")
        
        results_df = pd.DataFrame(self.results)
        
        # 按参数组合分组并计算平均指标
        param_names = list(self.param_grid.keys())
        summary = results_df.groupby(param_names)[metric].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        # 找出最优参数组合
        best_idx = summary['mean'].idxmax()
        best_params = {param: summary.loc[best_idx, param] for param in param_names}
        
        print(f"\nBest parameters: {best_params}")
        print(f"Mean {metric}: {summary.loc[best_idx, 'mean']:.4f} ± {summary.loc[best_idx, 'std']:.4f}")
        
        return best_params, summary
    
    def plot_results(self, metric: str = 'learning_agent_profit', output_file: str = None):
        """
        绘制结果图表
        
        Args:
            metric: 用于评估的指标
            output_file: 输出文件路径，如果为None则显示图表
        """
        if not self.results:
            raise ValueError("No results available. Run the tuner first.")
        
        results_df = pd.DataFrame(self.results)
        param_names = list(self.param_grid.keys())
        
        # 如果只有一个参数，绘制箱线图
        if len(param_names) == 1:
            param_name = param_names[0]
            plt.figure(figsize=(10, 6))
            ax = plt.subplot(111)
            results_df.boxplot(column=metric, by=param_name, ax=ax)
            plt.title(f'Effect of {param_name} on {metric}')
            plt.suptitle('')  # 移除默认标题
            plt.xlabel(param_name)
            plt.ylabel(metric)
            plt.tight_layout()
        
        # 如果有两个参数，绘制热力图
        elif len(param_names) == 2:
            param1, param2 = param_names
            pivot = results_df.pivot_table(
                index=param1, 
                columns=param2, 
                values=metric,
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 8))
            ax = plt.subplot(111)
            im = ax.imshow(pivot.values, cmap='viridis')
            
            # 设置坐标轴
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticklabels(pivot.index)
            
            # 添加颜色条和标题
            plt.colorbar(im, ax=ax)
            plt.title(f'Effect of {param1} and {param2} on {metric}')
            plt.xlabel(param2)
            plt.ylabel(param1)
            
            # 在每个单元格中添加数值
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    text = ax.text(j, i, f"{pivot.iloc[i, j]:.2f}",
                                  ha="center", va="center", color="w")
            
            plt.tight_layout()
        
        # 如果有更多参数，绘制平行坐标图
        else:
            # 计算每个参数组合的平均指标
            summary = results_df.groupby(param_names)[metric].mean().reset_index()
            
            # 标准化参数值以便绘图
            for param in param_names:
                if summary[param].dtype in [np.float64, np.int64]:
                    summary[f"{param}_norm"] = (summary[param] - summary[param].min()) / \
                                             (summary[param].max() - summary[param].min())
                else:
                    # 对于分类参数，使用类别编码
                    categories = summary[param].unique()
                    category_map = {cat: i/(len(categories)-1) for i, cat in enumerate(categories)}
                    summary[f"{param}_norm"] = summary[param].map(category_map)
            
            # 标准化指标
            summary[f"{metric}_norm"] = (summary[metric] - summary[metric].min()) / \
                                      (summary[metric].max() - summary[metric].min())
            
            # 绘制平行坐标图
            plt.figure(figsize=(12, 6))
            
            # 创建坐标轴
            param_norm_names = [f"{param}_norm" for param in param_names]
            param_norm_names.append(f"{metric}_norm")
            
            # 绘制每个参数组合的线
            for i, row in summary.iterrows():
                # 颜色基于指标值
                color = plt.cm.viridis(row[f"{metric}_norm"])
                
                # 绘制线
                plt.plot(range(len(param_norm_names)), [row[col] for col in param_norm_names], 
                         color=color, alpha=0.7)
            
            # 设置坐标轴
            plt.xticks(range(len(param_norm_names)), param_names + [metric])
            plt.ylim(0, 1)
            
            # 添加颜色条
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                      norm=plt.Normalize(vmin=summary[metric].min(), 
                                                        vmax=summary[metric].max()))
            sm.set_array([])
            plt.colorbar(sm, label=metric)
            
            plt.title(f'Parallel Coordinates Plot of Parameters vs {metric}')
            plt.tight_layout()
        
        # 保存或显示图表
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def main():
    """
    超参数搜索主函数
    """
    # 定义参数网格
    param_grid = {
        'RL_ALGORITHM': ['PPO', 'DDPG'],  # 强化学习算法
        'w_profit': [0.5, 1.0, 1.5],  # 利润奖励权重
        'w_pacing': [0.1, 0.2, 0.3],  # 预算节奏奖励权重
        'w_efficiency': [0.2, 0.3, 0.4]  # 成本效率奖励权重
    }
    
    # 创建超参数搜索器
    tuner = HyperparameterTuner(
        param_grid=param_grid,
        n_trials=3,  # 每个参数组合重复3次
        output_dir="tuning_results"
    )
    
    # 运行超参数搜索
    results = tuner.run()
    
    # 分析结果
    best_params, summary = tuner.analyze_results()
    
    # 绘制结果
    tuner.plot_results(output_file="tuning_results/parameter_effects.png")
    
    print("\nHyperparameter tuning completed!")
    print(f"Results saved to {tuner.output_dir}")


if __name__ == "__main__":
    main()