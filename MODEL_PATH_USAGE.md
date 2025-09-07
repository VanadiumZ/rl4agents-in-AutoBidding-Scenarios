# 模型路径使用说明

## 概述

修改后的 `run_with_model.py` 现在接受 `--model_path` 参数，允许用户直接指定模型文件的相对路径。

## 使用方法

### 基本语法
```bash
python run_with_model.py --model_path <相对路径> --type <模型类型>
```

### 示例用法

#### 1. 加载BC模型
```bash
python run_with_model.py --model_path auction_sim/models/bc_ippo/best/Learning_0_ppo_model.pth --type bc
```

#### 2. 加载MADDPG模型
```bash
python run_with_model.py --model_path auction_sim/models/bc_maddpg/best/Learning_0_actor.pth --type maddpg
```

#### 3. 加载其他BC模型
```bash
python run_with_model.py --model_path auction_sim/models/bc_direct/best/Learning_0_model.pth --type bc
```

#### 4. 交互式选择（无需指定路径）
```bash
python run_with_model.py
```

#### 5. 列出可用模型
```bash
python run_with_model.py --list
```

## 支持的模型类型

- `bc`: 行为克隆模型
- `ppo`: PPO模型
- `mappo`: Multi-Agent PPO模型
- `maddpg`: Multi-Agent DDPG模型
- `auto`: 自动检测（默认）

## 模型文件结构

```
auction_sim/models/
├── bc_ippo/best/Learning_0_ppo_model.pth
├── bc_maddpg/best/Learning_0_actor.pth
├── bc_maddpg/best/Learning_0_critic.pth
├── bc_direct/best/Learning_0_model.pth
└── ...
```

## 注意事项

1. 路径必须是相对于项目根目录的相对路径
2. 确保指定的模型文件存在
3. 模型类型必须与实际模型文件匹配
4. BC模型现在直接加载指定的文件，不再在目录中搜索