#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型加载功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from auction_sim.model_runner import load_trained_model

def test_model_loading():
    """
    测试不同类型模型的加载
    """
    print("🧪 开始测试模型加载功能...\n")
    
    # 测试BC模型加载
    bc_model_path = "auction_sim/models/bc_ippo/best/Learning_0_ppo_model.pth"
    if os.path.exists(bc_model_path):
        try:
            print(f"📁 测试BC模型: {bc_model_path}")
            trainer, model_type = load_trained_model(bc_model_path, "bc")
            print(f"✅ BC模型加载成功，类型: {model_type}")
        except Exception as e:
            print(f"❌ BC模型加载失败: {e}")
    else:
        print(f"⚠️  BC模型文件不存在: {bc_model_path}")
    
    print()
    
    # 测试MADDPG模型加载
    maddpg_actor_path = "auction_sim/models/bc_maddpg/best/Learning_0_actor.pth"
    if os.path.exists(maddpg_actor_path):
        try:
            print(f"📁 测试MADDPG Actor模型: {maddpg_actor_path}")
            trainer, model_type = load_trained_model(maddpg_actor_path, "maddpg")
            print(f"✅ MADDPG模型加载成功，类型: {model_type}")
        except Exception as e:
            print(f"❌ MADDPG模型加载失败: {e}")
    else:
        print(f"⚠️  MADDPG模型文件不存在: {maddpg_actor_path}")
    
    print("\n🎯 模型加载测试完成！")

if __name__ == "__main__":
    test_model_loading()