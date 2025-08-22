<div style="text-align: center;">
    项目名称：智能体强化学习在自动出价场景下的应用
    <br>
    The Application of Reinforcement Learning for Agents in Automated Bidding Scenarios
</div>

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

### **1\. 项目背景与目标**

在在线广告领域，多个广告商（Agents）需要对同一个广告位进行竞价。谁的出价高，谁的广告就排在前面。这个过程充满了不确定性：

* **对手的不确定性：** 你不知道竞争对手会出多少价。  
* **价值的不确定性：** 你不完全确定一次点击到底能给你带来多少真实收益（Click-Through Rate, CTR，即点击率，和 Conversion Rate，即转化率，都是估算的）。

**项目目标：**在一个简化的、模拟的广告竞价环境中，创建多个具有不同出价策略的智能体（Agent），并观察和分析在长期竞争中，哪种策略能获得最高的**综合经济价值（Economic Value）**。

$$
ROI _{Agent}  =  \left(\frac{该Agent的累计利润} {该Agent的累计支付成本}\right)×100%
\\
Profit_{Agent}  =  \sum_{t=  1}^T(TrueValue_t \times CTR_{pos} - Cost_t)
\\
WinRate_{Agent} = \frac{该Agent的获胜次数}{总拍卖轮次}
\\
TargetWins = \frac{总拍卖轮次}{智能体总数} \times 0.8 \quad \text{(期望竞争获胜次数)}
\\
\textbf{Agents' Target: } \max \left( 0.5 \cdot Profit_{Agent} + 0.15 \cdot ROI_{Agent} \cdot TotalCost_{Agent} + 0.35 \cdot WinRate_{Agent} \cdot TargetWins \right)
$$

**目标函数解释：**
- **利润项 (50% 权重)**: 直接优化累计利润，鼓励盈利性出价
- **效率项 (15% 权重)**: ROI加权的投资回报，鼓励高效的资金使用
- **竞争力项 (35% 权重)**: 获胜率相对于期望表现，鼓励积极参与竞争

这种多目标优化避免了过度保守或过度激进的策略，促使智能体在盈利性、效率性和竞争力之间找到平衡。

### **2\. 核心概念与简化**

我们做以下约定和假设，以构建一个既可控又贴近现实的模拟环境。

* **拍卖机制：** 采用**广义第二价格拍卖 (Generalized Second-Price Auction, GSP)**。这是早期Google和百度广告系统使用的真实机制。简单来说，第一名赢得最好的位置，但只需要支付第二名的出价；第二名赢得次好的位置，支付第三名的出价，以此类推。
* **智能体（Agents）：** 代表不同的广告商。我们设计几种不同"性格"的智能体。  
* **环境（Environment）：** 一个模拟器，在每一轮，都发起一次拍卖。  
  - **广告位数量 (Number of Ad Slots):** 设定有 `N_slots = 4` 个广告位。这意味着每轮拍卖最多有4个赢家，出价排在 `N_slots` 之后的智能体将输掉本轮拍卖。增加到4个广告位可以让50%的智能体获胜，降低竞争压力。
  - **位置点击率 (Positional CTR):** 排名直接影响点击率。我们假设不同位置的CTR是固定的，且排名越高CTR越高。
    - **`CTR_positions = [0.8, 0.65, 0.5, 0.35]`** 分别对应第1、2、3、4名的位置。这是一个更平缓的递减曲线，更接近真实场景。具体到代码实施层面，为位置 CTR 加入轻微扰动(±5%)，模拟不同页面布局的影响
    - 因此，智能体的**期望收益**将是 `(感知价值 * 赢得位置的CTR) - 支付成本`。
* **预算约束 (Budget Constraint):** 每个智能体都拥有一个初始的**总预算** (`AGENT_BUDGET`，默认 30,000)。每次获胜后，预算将扣除相应成本，当余额不足以支付本轮出价时将自动弃权。
* **不确定性建模：**  
  1. **价值不确定性：** 假设每次点击的"真实价值"为一个基础值 $V$，$V$ 服从均匀分布 $U(10.0, 40.0)$（3倍扩展以匹配预算规模，确保正向ROI）。每个智能体观察到的价值是 $V' = V + \epsilon$，其中 $\epsilon$ 是正态分布的随机噪声 $N(0, 0.15 \times V)$（优化到15%噪声，提高决策精度）。这模拟了每个广告商对同一次点击有不同的价值评估。  
  2. **对手不确定性：** 这是天然存在的，因为智能体之间互相不知道对方的策略和出价。

### **3\. 项目实施步骤 (Methodology)**

#### **第一步：搭建模拟环境**

编写一个 Auction 类。

* Auction 类需要一个方法，比如 run\_auction(bids)。  
* 输入：一个包含所有智能体出价的列表或字典，例如 {'agent\_A': 1.2, 'agent\_B': 1.5, 'agent\_C': 1.35}。  
* 过程：  
  1. 对出价进行排序。  
  2. 根据GSP规则确定每个赢家的支付价格（pay-per-click, PPC）。
* 输出：拍卖结果，包括每个智能体的排名和需要支付的费用。

#### **第二步：定义不同策略的智能体**

创建一个 Agent 基类，然后派生出不同策略的子类。每个Agent都有自己的ID、预算（可选）、以及一个核心的 bid() 方法。

1. **"老实人"智能体 (Truthful Agent):**  
   * **策略：** 它认为这次点击值多少钱，就出多少价。  
   * **逻辑：** $bid\_price = perceived\_value$。这是最简单的基准策略。  
   
2. **"保守派"智能体 (Conservative Agent):**  
   
   * **策略：** 目标是在整个拍卖周期内(例如1000轮)平稳地花光预算，避免"钱到用时方恨少"或"周期结束钱没花完"。
   * **逻辑：**它会根据预算消耗的"进度"来动态调整出价的系数。
     1. 计算理想平均每轮花费(ldeal Pace): $IdealPace = \frac{InitialBudget}{TotalRounds}$
     2. 计算当前实际每轮花费(Actual Pace): $ActualPace = \frac{BudgetSpent}{CurrentRound}$
     3. 根据两个Pace对比生成一个动态调整因子$\alpha$:
        - 如果花得太慢(`ActualPace < IdealPace`)，就需要更激进一点，$\alpha$ 会大于1。
        - 如果花得太快(`ActualPace >IdealPace`)，就需要更保守一点，$\alpha$ 会小于1。
     4. 最终出价：$$bid\_price=perceived\_value \times \alpha$$，其中 $\alpha = f\left(\frac{IdealPace}{ActualPace}\right)$，可以用一个函数（甚至就是这个比率本身）使其变化更平滑。
   
3. **"激进派"智能体 (Aggressive Agent):**  
   
   * **策略：**智能体的激进程度由它最近的获胜率决定。胜率低就提高出价，胜率高就回归理性，非常符合直觉。
   * **逻辑：**智能体有一个期望的胜率 `target_win_rate = 1 / N_agents`。我们跟踪当前胜率：维护一个最近N轮(例如N=15)的胜率`current_win_rate`，于是我们可以计算**好胜因子$\beta$**：
     - 如果当前胜率低于目标，说明自己太"保守"了，需要更激进。$\beta$ 大于1。
     - 如果当前胜率高于目标，说明最近有点"上头"，可以稍微冷静一下。$\beta$ 会约等于1。
     - **最终出价**：$$bid\_price=perceived\_value \times \beta = perceived\_value \times (1.0 + \lambda(target\_win\_rate - current\_win\_rate))$$
   
4. **"自适应"学习智能体 (Simple Adaptive Agent):**  
   * **策略：** **这是项目的核心！**这需要我们定义强化学习的核心三要素：**状态（State）、动作（Action）、奖励（Reward）**。
   
   * **强化学习问题定义 (MDP Formulation)**
   
     - **状态 (State, S)：** 
   
       - `[perceived_value]`
       - ``perceived_value_t`: 当前轮次的感知价值。
       - `remaining_budget_ratio`: 剩余预算 / 初始预算。这对于预算控制至关重要。
       - `time_ratio`: 剩余轮次 / 总轮次。这告诉智能体时间的紧迫性。
       - `recent_win_rate`: 最近 N 轮的胜率。反映了近期的竞争激烈程度。
       - `recent_avg_profit_per_win`: 最近 N 轮平均每次获胜的利润。反映了出价是否"过高"。
   
     - **动作 (Action, A)：** **连续动作空间：** 动作直接是出价金额 `bid`。Actor的输出通常是经过 `tanh` 激活函数的，范围在 `[-1, 1]` 之间。我们必须必须将其缩放到一个有意义的出价范围！
   
     - **奖励 (Reward, R)：** 
   
       - 这里很Tricky，Reward设计对它而言至关重要!!!
   
       - **方案一：直接利润奖励**
   
         $$R_t=Profit_t=(PerceivedValuet\times CTR_{won_position}−Cost_t)$$，**如果失败：** 奖励为 0。
   
       - **方案二：塑造成形的复合奖励**
   
         $$R_t=Profit_t + w_1 \times R_{pacing, t} + w_2 \times R_{efficiency, t}$$，
   
         其中 $w$ 是各项的权重，用于平衡不同目标。
   
         1. **预算节奏奖励 (Rpacing):** 惩罚那些花钱太快或太慢的行为。
            - 定义理想花费速率：`ideal_spend_per_round = initial_budget / total_rounds`
            - 计算当轮花费与理想值的差距：`overspend = Cost_t - ideal_spend_per_round`
            - **奖励设计：** 可以设计一个惩罚项，比如 $R_{pacing,t}=-\max(0, overspend)$。
            - **目的：** 鼓励智能体像"保守派"一样，学会有节奏地花费预算。
         2. **成本效率奖励 (Refficiency):** 直接在单步奖励中体现 ROI 的思想。
            - **奖励设计：** $R_{efficiency,t}=\frac{Profit_{t}}{Cost_t + \epsilon}$ (其中 ϵ 是一个极小值如 1e-8)。
            - **目的：** 直接鼓励智能体在"赚钱"的同时，还要"赚得漂亮"，即花小钱办大事。
            - **注意：** 这个奖励项的方差可能很大（Cost 很小时，该值会剧烈波动），可能会让训练变得不稳定。使用时需要谨慎调整其权重 $w_{efficiency}$，通常会设置得比较小。

#### **第三步：运行模拟**

运行 N 轮拍卖。

* **在每一轮：**  
  1. 环境生成一个基础的"真实价值" V。  
  2. 为每个智能体生成带有噪声的"感知价值" V'。  
  3. 每个智能体调用自己的 bid() 方法，提交出价。  
  4. Auction 类运行拍卖，返回结果。  
  5. 为每个智能体计算本轮的利润（$profit = perceived\_value - cost$，如果输了则利润为0）。  
  6. 记录每个智能体的累计利润、获胜次数等数据。  
  7. "自适应"智能体根据结果更新其出价系数 c。

#### **第四步：结果分析与可视化**

**累计利润 vs. 轮次**

- 每策略画累计利润曲线，热身期可用虚线表示不记录区间。

**ROI vs. 轮次**

- 每策略画ROI曲线，热身期可用虚线表示不记录区间。

**胜率与平均 CPC**

- 表格／条形图比较各策略胜率（赢得至少一个位的比例）和平均每次胜利的支付价格。

**预算耗尽时间分布**

- KDE／直方图展示各策略预算耗尽的轮次，反映耗预算速度。 

**胜率分析：** 

- 统计每个智能体的总获胜次数，计算胜率。  

### 4\. 实验具体要求

设"自适应"学习智能体的数目为 $k$，当 $k = 0, 1, 2$ 时，设置 `AGENT_BUDGET` 分别为 `[20000.0, 30000.0, 40000.0]` 时，完成以下实验：

`["老实人"智能体，"激进派"智能体，"保守派"智能体]` 数目分别为 `[0, 2, 2], [2, 0, 2], [2, 2, 0], [2, 2, 2]` 时的实验结果。

**请注意**："老实人"智能体，"激进派"智能体，"保守派"智能体都是确定性的智能体，只有"自适应"学习智能体是我们强化学习要学习的目标，$k = 0$ 是对照实验，$k = 1$ 是单智能体强化学习，$k=2$ 时该项目变成多智能体强化学习。



k = 1

DDPG、PPO


k = 2

IPPO、MAPPO/MADDPG



```python
/rl4agents-in-AutoBidding-Scenarios
│  README.md
│  run.py
│
└─auction_sim
    │  agents.py		# 各类 Agent 定义
    │  auction.py		# 拍卖环境封装
    │  config.py		# 超参数与实验设置
    │  runner.py		# 主流程调度
    └── results/        # 实验数据与图表输出

    class Agent:
    def __init__(self, id, budget, noise_std):
        self.id = id
        self.budget = budget
        self.noise_std = noise_std
    def perceive(self, true_value):
        # 返回含噪价值 V'
    def bid(self, perceived_value) -> float:
        # 子类实现
    def update(self, result):
        # 自适应策略重写，其他默认不变
  
TruthfulAgent, ConservativeAgent(k), AggressiveAgent(m)
```

### **5\. 现有基础代码一览（Quick Code Tour）**

| 目录/文件 | 作用简述 |
|-----------|---------|
| `auction_sim/agents.py` | ① 定义抽象基类 `Agent`（统一接口：`perceive()` / `bid()` / `update()`）<br/>② 已实现三种规则智能体：`TruthfulAgent`、`ConservativeAgent`、`AggressiveAgent`<br/>③ `LearningAgent` 框架留好接口，等待接入 PPO / DDPG 等强化学习算法 |
| `auction_sim/auction.py` | 实现 **广义第二价格拍卖** (`GSPAuction`)：排序、扣费、带噪 CTR 计算 |
| `auction_sim/runner.py` | 主模拟脚本：解析 `config.py` → 创建智能体 → 循环 N 轮拍卖 → 收集并打印结果 |
| `auction_sim/config.py` | 全局超参（拍卖轮数、CTR、**`AGENT_BUDGET` 默认 30 000**、各类智能体数量…）——改这里即可做不同实验 |
| `run.py` | 入口占位，等价于 `python -m auction_sim.runner` |

**快速开始**

```bash
# 安装依赖
pip install numpy tqdm

# 运行基准模拟（k=0 已在 config.py 中预设）
python -m auction_sim.runner
```
若要调整智能体组合/数量/预算，只需修改 `config.EXPERIMENT_SETUP`。

---

## **6. 学习智能体性能问题分析与解决方案**

### **6.1 问题诊断**

在初始实验中，学习智能体表现出以下问题：

- **极低胜率**（2-3%）：远低于理论值（50%）
- **预算利用率不足**（<5%）：过度保守的出价策略
- **陷入局部最优**：学会了极度保守以避免亏损

**根本原因**：
1. 多目标优化过于复杂
2. 稀疏奖励导致探索困难
3. 直接面对专家级对手，学习曲线过陡

### **6.2 解决方案：课程学习**

设计了9阶段渐进式课程学习方案，在`config.py`中实现为`CurriculumStage`：

```python
class CurriculumStage(Enum):
    STAGE_0 = 0  # 2学习 + 2诚实
    STAGE_1 = 1  # 2学习 + 2保守 + 2诚实
    STAGE_2 = 2  # 2学习 + 2激进 + 2保守 + 2诚实
    ...
    STAGE_8 = 8  # 完整8智能体环境
```

每个阶段逐步增加对手数量和复杂度，使用统一的最终奖励函数：
```
reward = 0.5 × Profit + 0.15 × ROI × TotalCost + 0.35 × WinRate × TargetWins
```

### **6.3 行为克隆(BC)预训练**

通过模仿Truthful智能体快速建立基础策略：
1. 收集10 episodes数据（~1600个样本）
2. 监督学习训练Actor网络（10 epochs）
3. 达到99.9%的动作预测准确率

**效果**：显著加速收敛，避免早期探索困难。

---

## **7. 实验结果**

### **7.1 训练方法对比**

| 方法 | 训练时间 | 平均胜率 | 平均ROI | 推荐指数 |
|------|----------|----------|---------|----------|
| **纯BC** | 5分钟 | 20.0% | 68.0% | ⭐⭐ |
| **纯课程学习** | 37小时 | 21.8% | 97.2% | ⭐⭐⭐ |
| **BC+直接训练** | 46分钟 | 18.9% | 66.8% | ⭐⭐⭐⭐⭐ |

**关键发现**：BC+直接训练在效率和性能间取得最佳平衡，避免了复杂的阶段管理。

### **7.2 多智能体算法对比**

基于100 episodes完整评估（使用`evaluate_models.py`）：

| 算法 | L0胜率 | L1胜率 | 平均胜率 | 胜率差异 | L0_ROI | L1_ROI | 平均ROI | 训练时间 |
|------|--------|--------|----------|----------|--------|--------|---------|----------|
| **MAPPO** | 25.1% | 10.0% | 17.6% | 15.0% | 105.7% | 7.0% | 56.3% | 46分钟 |
| **IPPO** | 9.7% | 23.2% | 16.5% | **13.6%** | 3.1% | 81.9% | 42.5% | 48分钟 |
| **MADDPG** | **92.2%** | 19.8% | **56.0%** | 72.4% | **921.5%** | 60.3% | **490.9%** | 2.5小时 |

### **7.3 算法特性分析**

| 特性 | MAPPO | IPPO | MADDPG |
|------|-------|------|--------|
| **架构** | 独立网络* | 完全独立 | Actor-Critic分离 |
| **稳定性** | 中等 | **最好** | 较差 |
| **不对称性** | 中等 | **最低** | 最高 |
| **训练效率** | 快速 | 快速 | 缓慢 |
| **推荐指数** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

*注：虽然名为MAPPO，但实际实现中每个智能体有独立网络

---

## **8. 核心结论**

### **8.1 关键发现**

1. **BC预训练的关键作用**：解决了"不敢出价"的根本问题，使后续优化更高效
2. **性能vs公平性权衡**：MADDPG性能最高但不对称性严重（72.4%差异）
3. **IPPO最均衡**：虽然平均性能不是最高，但智能体间差异最小（13.6%）

### **8.2 最终推荐**

> **BC预训练 + IPPO是目前最均衡的多智能体训练方案**，在保持训练稳定性的同时最好地解决了不对称问题。MADDPG虽然平均性能最高（56%胜率，490.9% ROI），但严重的不对称性使其不适合需要公平性的多智能体场景。

### **8.3 改进方向**

1. **MADDPG优化**：
   - 降低tau值减缓目标网络更新
   - 引入对称性正则化项
   - 调整探索噪声参数

2. **MAPPO改进**：
   - 实现真正的参数共享版本
   - 添加智能体ID嵌入
   
3. **IPPO增强**：
   - 增加网络容量提升性能上限
   - 尝试部分参数共享

---

## **9. 项目结构与使用**

### **9.1 完整项目结构**

```
rl4agents-in-AutoBidding-Scenarios/
├── auction_sim/
│   ├── agents.py              # 智能体基类与规则智能体
│   ├── auction.py             # GSP拍卖机制
│   ├── config.py              # 环境配置与课程设置
│   ├── ma_environment.py      # 多智能体环境
│   ├── ma_trainer.py          # MAPPO训练器
│   ├── ippo_trainer.py        # IPPO训练器
│   └── models/                # 保存的模型
├── bc_direct_trainer.py        # BC+MAPPO训练
├── bc_ippo_trainer.py          # BC+IPPO训练
├── bc_maddpg_trainer.py        # BC+MADDPG训练
├── pure_curriculum_trainer.py  # 纯课程学习
├── evaluate_models.py          # 模型评估脚本
└── README.md
```

### **9.2 快速开始**

```bash
# 安装依赖
pip install numpy torch matplotlib tqdm

# 训练模型（推荐）
python bc_ippo_trainer.py

# 评估模型
python evaluate_models.py --algorithm ippo --episodes 100

# 对比所有算法
python evaluate_models.py
```

### **9.3 配置说明**

在`config.py`中调整关键参数：
- `N_ROUNDS`：拍卖轮数（默认1600）
- `AGENT_BUDGET`：初始预算（默认30000）
- `CTR_POSITIONS`：广告位CTR（默认[0.8, 0.65, 0.5, 0.35]）
- `TRUE_VALUE_RANGE`：真实价值范围（默认[10.0, 40.0]）

---

## **10. 致谢与引用**

本项目为多智能体强化学习在广告竞价领域的探索性研究。如使用本代码，请引用：

```bibtex
@misc{marl_autobidding2024,
  title={Multi-Agent Reinforcement Learning for Auto-Bidding in Online Advertising},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/rl4agents-in-AutoBidding-Scenarios}
}
```

欢迎贡献代码和提出改进建议！