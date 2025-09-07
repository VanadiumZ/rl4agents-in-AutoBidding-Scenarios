# 模型使用指南

本指南介绍如何使用训练好的模型进行拍卖模拟。

## 📁 文件说明

- `model_runner.py` - 核心模型运行模块
- `run_with_model.py` - 完整功能的外层运行脚本（支持命令行参数和交互式选择）
- `run_model.py` - 简化版运行脚本（自动查找最佳模型）

## 🚀 快速开始

### 方法1: 简化版运行（推荐）

```bash
# 自动查找并运行最佳可用模型
python run_model.py
```

这个脚本会自动按优先级查找以下模型：
1. `auction_sim/models/best` (MAPPO最佳模型)
2. `auction_sim/models/final` (MAPPO最终模型)
3. `auction_sim/models/mappo` (MAPPO模型)
4. `auction_sim/models/ppo` (PPO模型)
5. `auction_sim/bc_model.pth` (BC预训练模型)
6. `auction_sim/bc_direct_model.pth` (BC Direct模型)

### 方法2: 交互式选择

```bash
# 交互式选择可用模型
python run_with_model.py
```

程序会列出所有可用模型，让你选择要使用的模型。

### 方法3: 命令行指定模型

```bash
# 使用MAPPO模型
python run_with_model.py --model auction_sim/models/best --type mappo

# 使用BC模型
python run_with_model.py --model auction_sim/bc_model.pth --type bc

# 使用PPO模型
python run_with_model.py --model auction_sim/models/ppo --type ppo

# 自定义模拟轮数
python run_with_model.py --model auction_sim/models/best --rounds 1000

# 列出所有可用模型
python run_with_model.py --list
```

## 📊 支持的模型类型

### 1. BC (Behavioral Cloning) 模型
- **文件格式**: `.pth` 文件
- **示例路径**: `auction_sim/bc_model.pth`
- **说明**: 行为克隆预训练模型，模仿专家智能体行为

### 2. PPO (Proximal Policy Optimization) 模型
- **文件格式**: 包含多个 `*_ppo_model.pth` 文件的目录
- **示例路径**: `auction_sim/models/ppo/`
- **说明**: 单智能体PPO强化学习模型

### 3. MAPPO (Multi-Agent PPO) 模型
- **文件格式**: 包含 `shared_model.pth` 的目录
- **示例路径**: `auction_sim/models/best/`
- **说明**: 多智能体PPO模型，使用共享网络架构

## 🔧 程序化调用

你也可以在Python代码中直接调用：

```python
from auction_sim.model_runner import run_simulation_with_trained_model

# 运行MAPPO模型模拟
agents = run_simulation_with_trained_model(
    model_path="auction_sim/models/best",
    model_type="mappo",
    n_rounds=1000
)

# 运行BC模型模拟
agents = run_simulation_with_trained_model(
    model_path="auction_sim/bc_model.pth",
    model_type="bc"
)

# 自动检测模型类型
agents = run_simulation_with_trained_model(
    model_path="path/to/model",
    model_type="auto"
)
```

## 📈 输出结果

运行完成后，程序会输出：

1. **控制台统计信息**:
   - 每个智能体的预算剩余
   - 总成本和胜场数
   - 累计利润和ROI

2. **可视化图表** (保存到 `auction_sim/results/`):
   - 智能体性能对比图
   - 利润变化曲线
   - 出价行为分析
   - 胜率统计图

## 🛠️ 故障排除

### 常见问题

1. **找不到模型文件**
   ```
   ❌ 模型路径不存在: auction_sim/models/best
   ```
   **解决方案**: 确保已经运行过训练脚本生成模型文件

2. **依赖包缺失**
   ```
   ModuleNotFoundError: No module named 'torch'
   ```
   **解决方案**: 安装必要的依赖包
   ```bash
   pip install torch matplotlib pandas scipy numpy tqdm
   ```

3. **模型加载失败**
   ```
   RuntimeError: Error(s) in loading state_dict
   ```
   **解决方案**: 检查模型文件是否完整，或尝试重新训练模型

### 调试模式

如果遇到问题，可以在Python中逐步调试：

```python
from auction_sim.model_runner import load_trained_model

# 测试模型加载
try:
    model_trainer, model_type = load_trained_model("auction_sim/models/best", "auto")
    print(f"✅ 模型加载成功: {model_type}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
```

## 📝 配置说明

模拟参数在 `auction_sim/config.py` 中配置：

- `SIMULATION_ROUNDS`: 模拟轮数
- `AGENT_BUDGET`: 智能体初始预算
- `N_SLOTS`: 广告位数量
- `TRUE_VALUE_RANGE`: 真实价值范围

## 🔄 与训练脚本的关系

| 训练脚本 | 生成的模型 | 使用方法 |
|---------|-----------|----------|
| `bc_direct_trainer.py` | BC + MAPPO模型 | `run_model.py` 或指定路径 |
| `ma_runner.py` | MAPPO模型 | `run_with_model.py --type mappo` |
| `ppo_trainer.py` | PPO模型 | `run_with_model.py --type ppo` |
| `bc_trainer.py` | BC模型 | `run_with_model.py --type bc` |

现在你可以轻松地使用任何训练好的模型进行拍卖模拟了！🎉