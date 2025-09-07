#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
外层运行脚本 - 使用训练好的模型进行拍卖模拟
支持命令行参数和交互式选择
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径到sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from auction_sim.model_runner import run_simulation_with_trained_model, load_trained_model

def find_available_models(models_dir="auction_sim/models"):
    """
    查找可用的模型文件
    
    Returns:
        dict: 模型类型到路径的映射
    """
    available_models = {}
    
    if not os.path.exists(models_dir):
        print(f"⚠️  模型目录不存在: {models_dir}")
        return available_models
    
    # 遍历models下的每个子目录
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            best_dir = os.path.join(subdir_path, "best")
            if os.path.exists(best_dir):
                # 检查best目录中的模型文件
                best_files = os.listdir(best_dir)
                
                # 查找MAPPO模型 (shared_model.pth)
                if "shared_model.pth" in best_files:
                    available_models[f"MAPPO ({subdir})"] = best_dir
                
                # 查找PPO模型 (_ppo_model.pth)
                ppo_files = [f for f in best_files if f.endswith("_ppo_model.pth")]
                if ppo_files:
                    available_models[f"PPO ({subdir})"] = best_dir
                
                # 查找MADDPG模型 (actor.pth和critic.pth)
                actor_files = [f for f in best_files if f.endswith("_actor.pth")]
                critic_files = [f for f in best_files if f.endswith("_critic.pth")]
                if actor_files and critic_files:
                    available_models[f"MADDPG ({subdir})"] = best_dir
                
                # 查找BC模型 (Learning_0_model.pth)
                if "Learning_0_model.pth" in best_files:
                    available_models[f"BC ({subdir})"] = best_dir
    
    # 查找根目录下的BC模型文件
    for file in os.listdir(models_dir):
        if file.endswith(".pth") and "bc" in file.lower():
            file_path = os.path.join(models_dir, file)
            available_models[f"BC ({file})"] = file_path
    
    # 查找auction_sim目录下的BC模型
    bc_files = ["bc_model.pth", "bc_direct_model.pth"]
    for bc_file in bc_files:
        bc_path = os.path.join("auction_sim", bc_file)
        if os.path.exists(bc_path):
            available_models[f"BC ({bc_file})"] = bc_path
    
    return available_models

def interactive_model_selection():
    """
    交互式模型选择
    
    Returns:
        tuple: (model_path, model_type)
    """
    print("\n" + "="*60)
    print("🤖 模型拍卖模拟器 - 交互式模式")
    print("="*60)
    
    # 查找可用模型
    available_models = find_available_models()
    
    if not available_models:
        print("❌ 未找到可用的训练模型")
        print("\n请确保以下路径存在训练好的模型:")
        print("  - auction_sim/models/best/ (MAPPO模型)")
        print("  - auction_sim/models/ppo/ (PPO模型)")
        print("  - auction_sim/bc_model.pth (BC模型)")
        return None, None
    
    print(f"\n📁 找到 {len(available_models)} 个可用模型:")
    model_list = list(available_models.items())
    
    for i, (model_name, model_path) in enumerate(model_list, 1):
        print(f"  {i}. {model_name}")
        print(f"     路径: {model_path}")
    
    # 用户选择
    while True:
        try:
            choice = input(f"\n请选择模型 (1-{len(model_list)}): ").strip()
            if choice.lower() in ['q', 'quit', 'exit']:
                print("👋 退出程序")
                return None, None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_list):
                selected_name, selected_path = model_list[choice_idx]
                print(f"\n✅ 已选择: {selected_name}")
                
                # 自动检测模型类型
                model_type = "auto"
                if "MAPPO" in selected_name:
                    model_type = "mappo"
                elif "PPO" in selected_name:
                    model_type = "ppo"
                elif "BC" in selected_name:
                    model_type = "bc"
                
                return selected_path, model_type
            else:
                print(f"❌ 请输入 1-{len(model_list)} 之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n👋 退出程序")
            return None, None

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description="使用训练好的模型进行拍卖模拟",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_with_model.py                                                    # 交互式选择模型
  python run_with_model.py --model_path auction_sim/models/bc_ippo/best/Learning_0_ppo_model.pth --type bc  # 使用BC模型
  python run_with_model.py --model_path auction_sim/models/mappo/best/shared_model.pth --type mappo         # 使用MAPPO模型
  python run_with_model.py --list                                             # 列出可用模型
        """
    )
    
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        help="模型文件的相对路径"
    )

    parser.add_argument(
        "--type", "-t",
        choices=["bc", "ppo", "mappo", "maddpg", "auto"],
        default="auto",
        help="模型类型 (默认: auto)"
    )
    
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        help="模拟轮数 (默认使用config中的设置)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有可用的模型"
    )
    
    args = parser.parse_args()
    
    # 列出可用模型
    if args.list:
        print("\n📁 可用模型列表:")
        available_models = find_available_models()
        if available_models:
            for model_name, model_path in available_models.items():
                print(f"  • {model_name}: {model_path}")
        else:
            print("  ❌ 未找到可用模型")
        return
    
    # 确定模型路径和类型
    if args.model_path:
        model_path = args.model_path
        model_type = args.type
        
        if not os.path.exists(model_path):
            print(f"❌ 模型路径不存在: {model_path}")
            return
        
        print(f"\n🚀 使用指定模型: {model_path}")
        print(f"📋 模型类型: {model_type}")
    else:
        # 交互式选择
        model_path, model_type = interactive_model_selection()
        if model_path is None:
            return
    
    # 运行模拟
    try:
        print(f"\n🎯 开始运行拍卖模拟...")
        if args.rounds:
            print(f"📊 模拟轮数: {args.rounds}")
        
        agents = run_simulation_with_trained_model(
            model_path=model_path,
            model_type=model_type,
            n_rounds=args.rounds
        )
        
        print(f"\n🎉 模拟完成！")
        print(f"📈 结果已保存到可视化图表中")
        
    except Exception as e:
        print(f"\n❌ 模拟过程中出现错误: {e}")
        print("\n🔧 可能的解决方案:")
        print("  1. 检查模型文件是否完整")
        print("  2. 确保所有依赖包已安装")
        print("  3. 检查config.py中的配置是否正确")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()