# /auction_sim/utils.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.stats import gaussian_kde
import os
from . import config

def create_results_dir():
    """创建结果目录"""
    results_dir = "auction_sim/results"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def plot_cumulative_profit(agents: List, save_path: str = None):
    """
    绘制累计利润vs轮次图
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, agent in enumerate(agents):
        # Skip agents with no history data
        if not agent.history:
            print(f"Warning: Agent {agent.id} has no history data, skipping in cumulative profit plot")
            continue
            
        # 计算每轮的累计利润
        cumulative_profits = []
        running_profit = 0
        
        for record in agent.history:
            running_profit += record.get('profit', 0.0)
            cumulative_profits.append(running_profit)
        
        if not cumulative_profits:  # Additional safety check
            print(f"Warning: Agent {agent.id} has no profit data, skipping")
            continue
        
        rounds = list(range(1, len(cumulative_profits) + 1))
        
        plt.plot(rounds, cumulative_profits, 
                label=f"{agent.id} (Final: {cumulative_profits[-1]:.2f})",
                color=colors[i % len(colors)], linewidth=2)
    
    plt.xlabel('Auction Rounds', fontsize=12)
    plt.ylabel('Cumulative Profit', fontsize=12)
    plt.title('Cumulative Profit vs. Rounds', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cumulative profit plot saved to: {save_path}")
    plt.show()

def plot_roi_over_time(agents: List, save_path: str = None):
    """
    绘制ROI vs轮次图
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, agent in enumerate(agents):
        # Skip agents with no history data
        if not agent.history or len(agent.history) < 100:
            print(f"Warning: Agent {agent.id} has insufficient history data for ROI plot, skipping")
            continue
            
        # 计算每轮的ROI（滑动窗口）
        window_size = 500  # 500轮滑动窗口
        roi_values = []
        rounds = []
        
        for end_idx in range(window_size, len(agent.history) + 1, 100):  # 每100轮计算一次
            window_records = agent.history[max(0, end_idx - window_size):end_idx]
            
            total_profit = sum(record.get('profit', 0.0) for record in window_records)
            total_cost = sum(record.get('cost', 0.0) for record in window_records)
            
            if total_cost > 0:
                roi = (total_profit / total_cost) * 100
                roi_values.append(roi)
                rounds.append(end_idx)
        
        if roi_values:  # 只有在有数据时才绘制
            plt.plot(rounds, roi_values, 
                    label=f"{agent.id} (Final ROI: {agent.get_roi():.2f}%)",
                    color=colors[i % len(colors)], linewidth=2)
    
    plt.xlabel('Auction Rounds', fontsize=12)
    plt.ylabel('ROI (%)', fontsize=12)
    plt.title('ROI vs. Rounds (500-round rolling window)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROI plot saved to: {save_path}")
    plt.show()

def plot_win_rate_and_cpc_analysis(agents: List, save_path: str = None):
    """
    绘制胜率与平均CPC分析
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    agent_names = [agent.id for agent in agents]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 计算统计数据
    win_rates = []
    avg_cpcs = []
    
    for agent in agents:
        # 胜率计算
        total_rounds = len(agent.history)
        # Handle different history formats - some have 'result' key, others have 'won' directly
        if agent.history and 'result' in agent.history[0]:
            wins = sum(1 for record in agent.history if record['result'] and record['result'].get('won', False))
        else:
            wins = sum(1 for record in agent.history if record.get('won', False))
        
        win_rate = (wins / total_rounds) * 100 if total_rounds > 0 else 0
        win_rates.append(win_rate)
        
        # 平均CPC计算（只计算获胜时的CPC）
        if agent.history and 'result' in agent.history[0]:
            winning_cpcs = [
                record['result']['cost_per_click'] 
                for record in agent.history 
                if record['result'] and record['result'].get('won', False)
            ]
        else:
            # For simulation agents, we don't have cost_per_click in the same format
            winning_cpcs = [
                record.get('cost', 0.0) 
                for record in agent.history 
                if record.get('won', False)
            ]
        
        avg_cpc = np.mean(winning_cpcs) if winning_cpcs else 0
        avg_cpcs.append(avg_cpc)
    
    # 绘制胜率条形图
    bars1 = ax1.bar(agent_names, win_rates, color=colors[:len(agents)])
    ax1.set_ylabel('Win Rate (%)', fontsize=12)
    ax1.set_title('Win Rate by Agent', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(win_rates) * 1.1)
    
    # 在条形图上添加数值标签
    for bar, rate in zip(bars1, win_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 绘制平均CPC条形图
    bars2 = ax2.bar(agent_names, avg_cpcs, color=colors[:len(agents)])
    ax2.set_ylabel('Average Cost-Per-Click', fontsize=12)
    ax2.set_title('Average CPC by Agent (Winning Auctions Only)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(avg_cpcs) * 1.1)
    
    # 在条形图上添加数值标签
    for bar, cpc in zip(bars2, avg_cpcs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{cpc:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Win rate and CPC analysis saved to: {save_path}")
    plt.show()

def plot_budget_depletion_distribution(agents: List, save_path: str = None):
    """
    绘制预算耗尽时间分布 - 改进版本
    """
    plt.figure(figsize=(15, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 计算每个智能体的预算消耗历史
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 预算随时间变化 (左上)
    for i, agent in enumerate(agents):
        rounds = [record['round'] for record in agent.history]
        budgets = [record['budget'] for record in agent.history]
        budget_ratios = [b / agent.initial_budget for b in budgets]
        
        ax1.plot(rounds, budget_ratios, 
                label=f"{agent.id}",
                color=colors[i % len(colors)], linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Budget Remaining (% of Initial)')
    ax1.set_title('Budget Depletion Over Time', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. 累计花费对比 (右上)
    agent_types = {}
    for agent in agents:
        agent_type = agent.id.split('_')[0]
        if agent_type not in agent_types:
            agent_types[agent_type] = []
        
        cumulative_costs = []
        running_cost = 0
        for record in agent.history:
            running_cost += record['cost']
            cumulative_costs.append(running_cost)
        
        agent_types[agent_type].append(cumulative_costs)
    
    # 绘制每种类型的平均花费
    for i, (agent_type, cost_histories) in enumerate(agent_types.items()):
        if cost_histories:
            # 计算平均值和标准差
            min_length = min(len(history) for history in cost_histories)
            aligned_histories = [history[:min_length] for history in cost_histories]
            mean_costs = np.mean(aligned_histories, axis=0)
            std_costs = np.std(aligned_histories, axis=0)
            rounds = list(range(1, min_length + 1))
            
            ax2.plot(rounds, mean_costs, 
                    label=f"{agent_type} (avg)", 
                    color=colors[i % len(colors)], linewidth=3)
            ax2.fill_between(rounds, 
                           np.maximum(0, mean_costs - std_costs),
                           mean_costs + std_costs,
                           alpha=0.2, color=colors[i % len(colors)])
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cumulative Cost')
    ax2.set_title('Average Cumulative Spending by Agent Type', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 预算耗尽轮次直方图 (左下)
    depletion_data = {}
    for agent in agents:
        threshold = agent.initial_budget * 0.05  # 5%阈值
        depletion_round = None
        
        for record in agent.history:
            if record['budget'] <= threshold:
                depletion_round = record['round']
                break
        
        agent_type = agent.id.split('_')[0]
        if agent_type not in depletion_data:
            depletion_data[agent_type] = []
        
        if depletion_round:
            depletion_data[agent_type].append(depletion_round)
        else:
            depletion_data[agent_type].append(config.SIMULATION_ROUNDS)
    
    # 绘制直方图
    all_depletion_rounds = []
    labels = []
    colors_hist = []
    
    for i, (agent_type, rounds) in enumerate(depletion_data.items()):
        all_depletion_rounds.extend(rounds)
        labels.extend([agent_type] * len(rounds))
        colors_hist.extend([colors[i % len(colors)]] * len(rounds))
    
    if all_depletion_rounds:
        # 按类型分组的直方图
        bins = np.linspace(0, config.SIMULATION_ROUNDS, 30)
        for i, (agent_type, rounds) in enumerate(depletion_data.items()):
            ax3.hist(rounds, bins=bins, alpha=0.7, 
                    label=f"{agent_type} (n={len(rounds)})",
                    color=colors[i % len(colors)], edgecolor='black')
    
    ax3.set_xlabel('Round when Budget ≤ 5% of Initial')
    ax3.set_ylabel('Count')
    ax3.set_title('Budget Depletion Time Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 花费效率对比 (右下)
    efficiency_data = []
    agent_names = []
    
    for agent in agents:
        total_cost = agent.get_total_cost()
        total_profit = agent.get_cumulative_profit()
        # Handle different history formats
        win_count = 0
        for record in agent.history:
            try:
                if 'result' in record and record['result'] and record['result'].get('won', False):
                    win_count += 1
                elif 'won' in record and record.get('won', False):
                    win_count += 1
            except (KeyError, TypeError):
                continue
        
        # 计算每次获胜的平均花费
        cost_per_win = total_cost / win_count if win_count > 0 else 0
        profit_per_cost = total_profit / total_cost if total_cost > 0 else 0
        
        efficiency_data.append({
            'agent': agent.id,
            'cost_per_win': cost_per_win,
            'profit_per_cost': profit_per_cost,
            'total_cost': total_cost
        })
    
    # 散点图: x=每次获胜成本, y=利润率, 气泡大小=总花费
    x_vals = [d['cost_per_win'] for d in efficiency_data]
    y_vals = [d['profit_per_cost'] for d in efficiency_data]
    sizes = [d['total_cost'] / 100 for d in efficiency_data]  # 缩放气泡大小
    
    scatter = ax4.scatter(x_vals, y_vals, s=sizes, alpha=0.7, c=range(len(agents)), cmap='tab10')
    
    # 添加标签
    for i, d in enumerate(efficiency_data):
        ax4.annotate(d['agent'], (d['cost_per_win'], d['profit_per_cost']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Cost per Win')
    ax4.set_ylabel('Profit per Cost')
    ax4.set_title('Spending Efficiency\n(Bubble size = Total Cost)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Improved budget analysis saved to: {save_path}")
    plt.show()

def generate_summary_table(agents: List, save_path: str = None):
    """
    生成并保存汇总表格
    """
    summary_data = []
    
    for agent in agents:
        total_cost = agent.get_total_cost()
        cumulative_profit = agent.get_cumulative_profit()
        roi = agent.get_roi()
        
        # Handle different history formats
        if agent.history and 'result' in agent.history[0]:
            win_count = sum(1 for record in agent.history if record['result'] and record['result'].get('won', False))
        else:
            win_count = sum(1 for record in agent.history if record.get('won', False))
        
        win_rate = (win_count / len(agent.history)) * 100 if agent.history else 0
        
        # 平均每次获胜的支付价格
        if agent.history and 'result' in agent.history[0]:
            winning_costs = [
                record['cost'] for record in agent.history 
                if record['result'] and record['result'].get('won', False)
            ]
        else:
            winning_costs = [
                record['cost'] for record in agent.history 
                if record.get('won', False)
            ]
        
        avg_cost_per_win = np.mean(winning_costs) if winning_costs else 0
        
        summary_data.append({
            'Agent_ID': agent.id,
            'Agent_Type': agent.id.split('_')[0],
            'Budget_Left': agent.budget,
            'Total_Cost': total_cost,
            'Win_Count': win_count,
            'Win_Rate(%)': win_rate,
            'Cumulative_Profit': cumulative_profit,
            'ROI(%)': roi,
            'Avg_Cost_Per_Win': avg_cost_per_win
        })
    
    df = pd.DataFrame(summary_data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Summary table saved to: {save_path}")
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False, float_format='%.2f'))
    print("="*80)
    
    return df

def generate_all_visualizations(agents: List):
    """
    生成所有可视化图表
    """
    results_dir = create_results_dir()
    
    print("Generating visualizations...")
    
    # 1. 累计利润图
    plot_cumulative_profit(agents, f"{results_dir}/cumulative_profit.png")
    
    # 2. ROI图
    plot_roi_over_time(agents, f"{results_dir}/roi_over_time.png")
    
    # 3. 胜率和CPC分析
    plot_win_rate_and_cpc_analysis(agents, f"{results_dir}/win_rate_cpc_analysis.png")
    
    # 4. 预算耗尽分布
    plot_budget_depletion_distribution(agents, f"{results_dir}/budget_depletion_distribution.png")
    
    # 5. 汇总表格
    generate_summary_table(agents, f"{results_dir}/experiment_summary.csv")
    
    print(f"\nAll visualizations saved to: {results_dir}/")
    print("Available files:")
    print("- cumulative_profit.png")
    print("- roi_over_time.png") 
    print("- win_rate_cpc_analysis.png")
    print("- budget_depletion_distribution.png")
    print("- experiment_summary.csv")