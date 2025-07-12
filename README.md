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

**项目目标：**在一个简化的、模拟的广告竞价环境中，创建多个具有不同出价策略的智能体（Agent），并观察和分析在长期竞争中，哪种策略能获得最高的**累计利润（Cumulative Profit）和投资回报率（ROI）**。
$$
ROI _{Agent}  =  \left(\frac{该Agent的累计利润} {该Agent的累计支付成本}\right)×100%
\\
Profit_{Agent}  =  \sum_{t=  1}^T(TrueValue_t \times CTR_{pos} - Cost_t)
\\
\textbf{Agents' Target: } \max \ ROI_{Agent} + Profit_{Agent}
$$

### **2\. 核心概念与简化**

我们做以下约定和假设，以构建一个既可控又贴近现实的模拟环境。

* **拍卖机制：** 采用**广义第二价格拍卖 (Generalized Second-Price Auction, GSP)**。这是早期Google和百度广告系统使用的真实机制。简单来说，第一名赢得最好的位置，但只需要支付第二名的出价；第二名赢得次好的位置，支付第三名的出价，以此类推。
* **智能体（Agents）：** 代表不同的广告商。我们设计几种不同“性格”的智能体。  
* **环境（Environment）：** 一个模拟器，在每一轮，都发起一次拍卖。  
  - **广告位数量 (Number of Ad Slots):** 设定有 `N_slots = 2` 个广告位。这意味着每轮拍卖最多有2个赢家，出价排在 `N_slots` 之后的智能体将输掉本轮拍卖。
  - **位置点击率 (Positional CTR):** 排名直接影响点击率。我们假设不同位置的CTR是固定的，且排名越高CTR越高。
    - **`CTR_positions = [0.7, 0.3]`** 分别对应第1、2名的位置。具体到代码实施层面，为位置 CTR 加入轻微扰动，比如每轮在原始值上加 ±5% 的随机小噪声，模拟不同页面布局的影响。
    - 因此，智能体的**期望收益**将是 `(感知价值 * 赢得位置的CTR) - 支付成本`。
* **预算约束 (Budget Constraint):** 每个智能体都拥有一个初始的**总预算** (`AGENT_BUDGET`，默认 30 000)。每次获胜后，预算将扣除相应成本，当余额不足以支付本轮出价时将自动弃权。README 推荐在三组实验中将 `AGENT_BUDGET` 分别设为 **20 000 / 30 000 / 40 000**，用来考察预算充裕度对策略表现的影响。（仍可单独为不同“性格”智能体指定预算，以模拟阔绰/拮据的广告主）
* **不确定性建模：**  
  1. **价值不确定性：** 假设每次点击的“真实价值”为一个基础值 $V$，$V$ 是一个均匀分布，这个 $V$ 不妨假定是一个均匀分布，但每个智能体观察到的价值是 $V' = V + \epsilon$，其中 $\epsilon$ 是一个小的随机噪声（例如，正态分布的随机数）。这模拟了每个广告商对同一次点击有不同的价值评估。  
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

1. **“老实人”智能体 (Truthful Agent):**  
   * **策略：** 它认为这次点击值多少钱，就出多少价。  
   * **逻辑：** $bid\_price = perceived\_value$。这是最简单的基准策略。  
   
2. **“保守派”智能体 (Conservative Agent):**  
   
   * **策略：** 目标是在整个拍卖周期内(例如1000轮)平稳地花光预算，避免“钱到用时方恨少“或“周期结束钱没花完”。
   * **逻辑：**它会根据预算消耗的“进度"来动态调整出价的系数。
     1. 计算理想平均每轮花费(ldeal Pace): $IdealPace = \frac{InitialBudget}{TotalRounds}$
     2. 计算当前实际每轮花费(Actual Pace): $ActualPace = \frac{BudgetSpent}{CurrentRound}$
     3. 根据两个Pace对比生成一个动态调整因子$\alpha$:
        - 如果花得太慢(`ActualPace < IdealPace`)，就需要更激进一点，$\alpha$ 会大于1。
        - 如果花得太快(`ActualPace >IdealPace`)，就需要更保守一点，$\alpha$ 会小于1。
     4. 最终出价：$$bid\_price=perceived\_value \times \alpha$$，其中 $\alpha = f\left(\frac{IdealPace}{ActualPace}\right)$，可以用一个函数（甚至就是这个比率本身）使其变化更平滑。
   
3. **“激进派”智能体 (Aggressive Agent):**  
   
   * **策略：**智能体的激进程度由它最近的获胜率决定。胜率低就提高出价，胜率高就回归理性，非常符合直觉。
   * **逻辑：**智能体有一个期望的胜率 `target_win_rate = 1 / N_agents`。我们跟踪当前胜率：维护一个最近N轮(例如N=15)的胜率`current_win_rate`，于是我们可以计算**好胜因子$\beta$**：
     - 如果当前胜率低于目标，说明自己太“保守”了，需要更激进。$\beta$ 大于1。
     - 如果当前胜率高于目标，说明最近有点“上头”，可以稍微冷静一下。$\beta$ 会约等于1。
     - **最终出价**：$$bid\_price=perceived\_value \times \beta = perceived\_value \times (1.0 + \lambda(target\_win\_rate - current\_win\_rate))$$
   
4. **“自适应”学习智能体 (Simple Adaptive Agent):**  
   * **策略：** **这是项目的核心！**这需要我们定义强化学习的核心三要素：**状态（State）、动作（Action）、奖励（Reward）**。
   
   * **强化学习问题定义 (MDP Formulation)**
   
     - **状态 (State, S)：** 
   
       - `[perceived_value]`
       - ``perceived_value_t`: 当前轮次的感知价值。
       - `remaining_budget_ratio`: 剩余预算 / 初始预算。这对于预算控制至关重要。
       - `time_ratio`: 剩余轮次 / 总轮次。这告诉智能体时间的紧迫性。
       - `recent_win_rate`: 最近 N 轮的胜率。反映了近期的竞争激烈程度。
       - `recent_avg_profit_per_win`: 最近 N 轮平均每次获胜的利润。反映了出价是否“过高”。
   
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
            - **目的：** 鼓励智能体像“保守派”一样，学会有节奏地花费预算。
         2. **成本效率奖励 (Refficiency):** 直接在单步奖励中体现 ROI 的思想。
            - **奖励设计：** $R_{efficiency,t}=\frac{Profit_{t}}{Cost_t + \epsilon}$ (其中 ϵ 是一个极小值如 1e-8)。
            - **目的：** 直接鼓励智能体在“赚钱”的同时，还要“赚得漂亮”，即花小钱办大事。
            - **注意：** 这个奖励项的方差可能很大（Cost 很小时，该值会剧烈波动），可能会让训练变得不稳定。使用时需要谨慎调整其权重 $w_{efficiency}$，通常会设置得比较小。

#### **第三步：运行模拟**

运行 N 轮拍卖。

* **在每一轮：**  
  1. 环境生成一个基础的“真实价值” V。  
  2. 为每个智能体生成带有噪声的“感知价值” V'。  
  3. 每个智能体调用自己的 bid() 方法，提交出价。  
  4. Auction 类运行拍卖，返回结果。  
  5. 为每个智能体计算本轮的利润（$profit = perceived\_value - cost$，如果输了则利润为0）。  
  6. 记录每个智能体的累计利润、获胜次数等数据。  
  7. “自适应”智能体根据结果更新其出价系数 c。

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

设“自适应”学习智能体的数目为 $k$，当 $k = 0, 1, 2$ 时，设置 `AGENT_BUDGET` 分别为 `[20000.0, 30000.0, 40000.0]` 时，完成以下实验：

`[“老实人”智能体，“激进派”智能体，“保守派”智能体]` 数目分别为 `[0, 2, 2], [2, 0, 2], [2, 2, 0], [2, 2, 2]` 时的实验结果。

**请注意**：“老实人”智能体，“激进派”智能体，“保守派”智能体都是确定性的智能体，只有“自适应”学习智能体是我们强化学习要学习的目标，$k = 0$ 是对照实验，$k = 1$ 是单智能体强化学习，$k=2$ 时该项目变成多智能体强化学习。



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
| `auction_sim/agents.py` | ① 定义抽象基类 `Agent`（统一接口：`perceive()` / `bid()` / `update()`）<br/>② 已实现三种规则智能体：`TruthfulAgent`、`ConservativeAgent`、`AggressiveAgent`<br/>③ `LearningAgent` 框架留好接口，等待接入 PPO / DDPG 等强化学习算法 |
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

### **6\. 实验路线图 / 合作分工建议**（以下内容主要由GPT-o3提供）

#### 6\.1 规则基线 (k = 0)
* **目标**：仅使用固定策略智能体，复现并分析拍卖动态，产出累计利润、ROI、预算消耗等图表。
* **待办**：
  1. 在 `runner.py` 完成真实利润与 ROI 统计；
  2. 新建 `auction_sim/utils.py`，封装日志与 `matplotlib` 绘图；
  3. 通过修改 `EXPERIMENT_SETUP` 跑完 README 中列出的四组组合，整理对比报告。

#### 6\.2 单智能体强化学习 (k = 1)
* **目标**：在规则对手环境下训练一个 `LearningAgent`（PPO / DDPG），验证其是否优于规则策略。
* **待办**：
  1. 完善 `LearningAgent.get_state()` 与 `scale_action()`；
  2. 选型并集成 RL 库（比如 [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3) 或者是 gym）；
  3. 在 `update()` 中收集 `(s, a, r, s')` 并调用 `model.learn()`；
  4. 超参搜索：奖励权重 \(w_1, w_2\)，学习率，折扣因子等；
  5. 记录并绘制训练曲线，撰写实验结论。

#### 6\.3 多智能体强化学习 (k = 2)
* **目标**：引入两名 `LearningAgent` 并探索合作 / 竞争学习（IPPO、MAPPO、MADDPG）。
* **待办**：
  1. 研究并选定多智能体算法框架，比如 `PettingZoo` API；
  2. 设计共享/独立状态与奖励；
  3. 调整 `runner.py` 使多智能体训练稳定（可能需要集中训练 + 分散执行的架构）；
  4. 对比单智能体与多智能体的收益、收敛性。

---

### **7\. 融合阿里 NeurIPS Auto Bidding Baseline 的高级实验（以下内容主要由GPT-o3提供）**

阿里天池比赛提供了更贴近真实广告场景的大规模模拟器与离线日志（仓库地址 <https://github.com/alimama-tech/NeurIPS_Auto_Bidding_General_Track_Baseline>）。我们计划在其代码基础上进行二次开发，步骤建议如下：

1. **环境对齐**
   * 克隆官方仓库至 `third_party/neurips2021_ab/`；
   * 阅读其 `environment.py` 与 `bid_agent.py`，确认拍卖接口、状态定义差异；
   * 编写 *适配层*，将本项目的 `Agent` 接口映射到阿里环境所需的 API。

2. **算法迁移**
   * 先在本轻量环境中调通并验证 RL 算法；
   * 再将经过调参的模型迁移到阿里环境做 *fine-tune* 或 *zero-shot* 测试。

3. **实验设计**
   * 对比以下三种训练策略：
     1. **Pure-Ali**：仅在阿里日志/模拟器上训练；
     2. **Pretrain-Our → Finetune-Ali**：先在本环境预训练，再迁移；
     3. **Curriculum**：逐渐增加环境复杂度（本项目 → 简化 Ali → 完整 Ali）。
   * 评估指标采用天池官方 `ROI`, `CPA`, `CTR` 等标准。

4. **结果提交 & 复现**
   * 使用 `wandb` / `tensorboard` 统一记录；
   * 提供 `scripts/run_ali.sh` 一键复现命令。

> **Tip**：阿里环境规模更大、状态更复杂，请务必先在本项目中验证思路，避免直接在大环境里摸黑调参导致时间成本过高。

