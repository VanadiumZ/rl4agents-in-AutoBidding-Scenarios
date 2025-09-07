#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版模型运行脚本
快速使用训练好的模型进行拍卖模拟
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from auction_sim.model_runner import run_simulation_with_trained_model

def main():
    """
    主函数 - 自动查找并运行最佳模型
    """
    print("🤖 模型拍卖模拟器")
    print("="*40)
    
    # 预定义的模型路径（按优先级排序）
    model_candidates = [
        ("auction_sim/models/best", "mappo", "MAPPO最佳模型"),
        ("auction_sim/models/final", "mappo", "MAPPO最终模型"),
        ("auction_sim/models/mappo", "mappo", "MAPPO模型"),
        ("auction_sim/models/ppo", "ppo", "PPO模型"),
        ("auction_sim/bc_model.pth", "bc", "BC预训练模型"),
        ("auction_sim/bc_direct_model.pth", "bc", "BC Direct模型"),
    ]
    
    # 查找第一个可用的模型
    selected_model = None
    for model_path, model_type, description in model_candidates:
        if os.path.exists(model_path):
            selected_model = (model_path, model_type, description)
            break
    
    if selected_model is None:
        print("❌ 未找到任何可用的训练模型")
        print("\n请确保以下路径之一存在训练好的模型:")
        for path, _, desc in model_candidates:
            print(f"  • {path} ({desc})")
        print("\n💡 提示: 请先运行训练脚本生成模型")
        return
    
    model_path, model_type, description = selected_model
    print(f"✅ 找到模型: {description}")
    print(f"📁 路径: {model_path}")
    print(f"🏷️  类型: {model_type.upper()}")
    
    try:
        print("\n🚀 开始运行拍卖模拟...")
        agents = run_simulation_with_trained_model(model_path, model_type)
        
        print("\n🎉 模拟完成！")
        print("📊 性能统计:")
        
        # 简单的性能统计
        learning_agents = [agent for agent in agents if "Learning" in agent.id]
        rule_agents = [agent for agent in agents if "Learning" not in agent.id]
        
        if learning_agents:
            learning_profits = [agent.get_cumulative_profit() for agent in learning_agents]
            learning_wins = [sum(1 for record in agent.history if record['result'] and record['result']['won']) for agent in learning_agents]
            
            print(f"  🧠 学习智能体平均利润: {sum(learning_profits)/len(learning_profits):.2f}")
            print(f"  🏆 学习智能体平均胜场: {sum(learning_wins)/len(learning_wins):.1f}")
        
        if rule_agents:
            rule_profits = [agent.get_cumulative_profit() for agent in rule_agents]
            rule_wins = [sum(1 for record in agent.history if record['result'] and record['result']['won']) for agent in rule_agents]
            
            print(f"  📏 规则智能体平均利润: {sum(rule_profits)/len(rule_profits):.2f}")
            print(f"  🎯 规则智能体平均胜场: {sum(rule_wins)/len(rule_wins):.1f}")
        
        print("\n📈 可视化图表已生成到 auction_sim/results/ 目录")
        
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        print("\n🔧 可能的解决方案:")
        print("  1. 检查模型文件是否完整")
        print("  2. 确保依赖包已安装: pip install torch matplotlib pandas scipy")
        print("  3. 检查config.py配置")

if __name__ == "__main__":
    main()