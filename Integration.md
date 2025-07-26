# 基于YuLan-OneSim的智能广告拍卖市场仿真

<div style="text-align: center;">
    项目名称：智能体语言推理在广告拍卖市场中的应用研究
    <br>
    Language-Reasoning Agents in Advertising Auction Markets: An Integration with YuLan-OneSim
</div>

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![YuLan-OneSim](https://img.shields.io/badge/YuLan--OneSim-v1.0-green.svg)
![Competition](https://img.shields.io/badge/RUC-AI智能体大赛-red.svg)

## **1. 项目背景与整合目标**

### **1.1 原项目特点**
我们的原始项目专注于**多智能体强化学习在自动出价场景下的应用**，通过MAPPO等算法训练智能体在GSP拍卖中进行策略优化。项目已实现：
- 完整的GSP拍卖环境
- 多种规则智能体（Truthful、Conservative、Aggressive）
- 强化学习智能体训练框架
- 多目标经济价值优化

### **1.2 YuLan-OneSim平台优势**
YuLan-OneSim为大语言模型智能体提供了强大的社会仿真框架：
- **语言推理能力**：智能体可以进行复杂的策略分析和决策推理
- **交互通信**：支持智能体间的语言交流和协商
- **场景构建**：通过JSON配置快速构建复杂社会场景
- **可视化界面**：提供直观的仿真过程展示和结果分析

### **1.3 整合目标**
**项目目标：** 将强化学习优化的拍卖策略转化为基于大语言模型的智能推理，探索**"数值优化"与"语言推理"两种智能体决策方式在拍卖市场中的表现差异**。

**研究问题：**
1. LLM智能体能否通过语言推理达到与RL智能体相当的策略水平？
2. 智能体间的语言交流如何影响拍卖市场的效率和公平性？
3. 不同推理策略（理性分析vs直觉判断）如何影响市场动态？

**技术创新点：**
- **混合智能体生态**：RL策略指导 + LLM语言推理的双重决策机制
- **策略传承机制**：将训练好的RL策略转化为LLM推理模板
- **市场分析智能体**：引入专门的市场分析师角色进行实时解读

## **2. 核心概念与适配方案**

### **2.1 智能体类型转换**

**原始智能体 → YuLan-OneSim智能体**

| 原始类型 | YuLan-OneSim适配 | 推理特征 | 决策方式 |
|---------|-----------------|---------|---------|
| **LearningAgent (MAPPO)** | **RationalAdvertiser** | 基于数据分析的理性决策 | 结合历史表现和市场分析 |
| **TruthfulAgent** | **HonestAdvertiser** | 诚实透明的价值评估 | 基于真实价值进行出价 |
| **ConservativeAgent** | **CautiousAdvertiser** | 风险厌恶的稳健策略 | 注重预算控制和长期收益 |
| **AggressiveAgent** | **CompetitiveAdvertiser** | 竞争导向的激进策略 | 基于胜率优化的动态调整 |
| **新增** | **MarketAnalyst** | 市场趋势分析和预测 | 提供市场洞察和建议 |

### **2.2 决策机制设计**

**2.2.1 理性决策模板（基于RL策略）**
```
作为一个理性的广告商，你需要综合考虑以下因素进行出价决策：

当前市场状态：
- 你的预算剩余：{budget_ratio:.2%}
- 拍卖剩余时间：{time_ratio:.2%} 
- 广告位价值：{perceived_value:.2f}
- 最近胜率：{recent_win_rate:.2%}
- 最近ROI：{recent_roi:.2%}

策略参考（基于强化学习优化结果）：
- 建议出价系数：{rl_suggestion:.3f}
- 历史最优范围：[{min_multiplier:.3f}, {max_multiplier:.3f}]
- 风险评估：{risk_level}

请基于以上信息进行策略分析，并给出你的最终决策。
```

**2.2.2 情感驱动决策模板**
```
你是一个{personality_type}的广告商，面临以下拍卖情况：

市场感知：
- 当前竞争激烈程度：{competition_level}
- 你的近期表现：{performance_trend}
- 对手动态：{competitor_analysis}

基于你的性格特征和情感倾向，你会如何应对这次拍卖？
考虑你的：风险偏好、决策风格、情感状态
```

### **2.3 环境配置规范**

**场景配置文件结构**
```
envs/intelligent_auction_market/
├── scene_info.json          # 场景描述和智能体类型定义
├── system_data_model.json   # 环境变量和智能体属性
├── actions.json             # 可执行动作定义
├── events.json              # 事件类型定义
├── profile/
│   ├── data/
│   │   ├── RationalAdvertiser.json     # 基于RL的理性智能体
│   │   ├── HonestAdvertiser.json       # 诚实型智能体
│   │   ├── CautiousAdvertiser.json     # 保守型智能体
│   │   ├── CompetitiveAdvertiser.json  # 竞争型智能体
│   │   └── MarketAnalyst.json          # 市场分析师
│   └── schema/              # 智能体属性模式定义
├── code/
│   ├── RationalAdvertiser.py    # 理性决策智能体实现
│   ├── PersonalityAdvertiser.py # 性格化智能体基类
│   ├── MarketAnalyst.py         # 市场分析智能体
│   ├── AuctionPlatform.py       # 拍卖平台（GSP机制）
│   └── events.py                # 自定义事件类型
└── workflow.html            # 仿真流程可视化
```

## **3. 实施步骤**

### **3.1 第一步：环境框架搭建**

**3.1.1 场景配置 (scene_info.json)**
```json
{
  "domain": "Economics",
  "scene_name": "intelligent_auction_market_simulation",
  "odd_protocol": {
    "overview": {
      "system_goal": "模拟大语言模型智能体在广告拍卖市场中的策略推理过程，研究语言推理与数值优化两种决策方式的差异，探索智能体交流对市场效率的影响。",
      "agent_types": "包含理性广告商（基于RL策略指导）、诚实广告商、谨慎广告商、竞争型广告商，以及市场分析师等角色。",
      "innovation_points": [
        "RL策略向语言推理的转化机制",
        "智能体间策略交流与学习",
        "多重决策模式的对比分析",
        "实时市场解读与预测"
      ]
    },
    "design_concepts": {
      "interaction_patterns": "智能体通过拍卖平台进行竞价，同时可以进行策略交流和市场讨论。",
      "decision_mechanisms": "结合历史数据分析、策略推理、情感判断和同伴学习的综合决策机制。"
    }
  }
}
```

**3.1.2 智能体属性定义 (system_data_model.json)**
```json
{
  "agents": {
    "RationalAdvertiser": {
      "variables": [
        {"name": "budget", "type": "float", "default_value": 30000.0},
        {"name": "risk_preference", "type": "str", "default_value": "balanced"},
        {"name": "rl_strategy_template", "type": "dict", "default_value": {}},
        {"name": "decision_reasoning", "type": "str", "default_value": ""},
        {"name": "bid_multiplier", "type": "float", "default_value": 1.0}
      ]
    },
    "MarketAnalyst": {
      "variables": [
        {"name": "analysis_reports", "type": "list", "default_value": []},
        {"name": "market_sentiment", "type": "str", "default_value": "neutral"},
        {"name": "trend_predictions", "type": "dict", "default_value": {}}
      ]
    }
  }
}
```

### **3.2 第二步：智能体实现**

**3.2.1 理性决策智能体**
```python
class RationalAdvertiser(GeneralAgent):
    """基于强化学习策略指导的理性决策智能体"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_event("AuctionStartEvent", "analyze_and_bid")
        self.register_event("MarketUpdateEvent", "update_strategy")
        
        # 加载预训练的RL策略参考
        self.rl_reference = self.load_rl_strategy_reference()
    
    async def analyze_and_bid(self, event: Event):
        # 获取市场状态
        market_state = self.get_market_state(event)
        
        # 获取RL策略建议
        rl_suggestion = self.get_rl_suggestion(market_state)
        
        instruction = f"""
        你是一个数据驱动的理性广告商，需要对当前拍卖进行策略分析。
        
        市场状态分析：
        - 当前预算使用率：{market_state['budget_ratio']:.2%}
        - 拍卖时间进度：{market_state['time_progress']:.2%}
        - 广告位预估价值：{market_state['perceived_value']:.2f}
        - 最近10轮胜率：{market_state['recent_win_rate']:.2%}
        - 最近ROI表现：{market_state['recent_roi']:.1f}%
        
        策略参考（基于历史最优表现）：
        - 建议出价系数：{rl_suggestion['multiplier']:.3f}
        - 预期胜率：{rl_suggestion['win_probability']:.2%}
        - 风险评估：{rl_suggestion['risk_level']}
        
        请基于数据分析进行决策推理，并给出具体的出价策略：
        {{
            "will_bid": <true/false>,
            "bid_multiplier": <float, 0.5-1.5>,
            "reasoning": "<你的分析推理过程>",
            "confidence": "<high/medium/low>",
            "expected_outcome": "<预期结果描述>"
        }}
        """
        
        result = await self.generate_reaction(instruction)
        return self.process_bidding_decision(result, event)
    
    def get_rl_suggestion(self, market_state):
        """基于预训练RL模型提供策略建议"""
        # 这里整合原项目的训练结果
        return {
            'multiplier': self.rl_reference.predict(market_state),
            'win_probability': self.estimate_win_probability(market_state),
            'risk_level': self.assess_risk_level(market_state)
        }
```

**3.2.2 市场分析智能体**
```python
class MarketAnalyst(GeneralAgent):
    """专门的市场分析师，提供实时市场解读"""
    
    async def analyze_market_trends(self, auction_history):
        instruction = f"""
        作为专业的广告拍卖市场分析师，请分析当前市场状况：
        
        最近拍卖数据：
        {self.format_auction_history(auction_history)}
        
        请提供：
        1. 市场趋势分析（价格走势、竞争激烈程度）
        2. 各类广告商策略识别
        3. 市场效率评估
        4. 对各智能体的策略建议
        
        输出格式：
        {{
            "market_trend": "<上涨/下跌/震荡>",
            "competition_level": "<激烈/适中/温和>",
            "efficiency_score": <0-100>,
            "strategy_recommendations": {{
                "conservative": "<建议>",
                "aggressive": "<建议>",
                "rational": "<建议>"
            }},
            "market_forecast": "<下阶段预测>"
        }}
        """
        
        analysis = await self.generate_reaction(instruction)
        await self.broadcast_market_report(analysis)
        return analysis
```

### **3.3 第三步：策略传承机制**

**3.3.1 RL策略知识提取**
```python
class StrategyTransferModule:
    """将训练好的RL策略转化为LLM推理模板"""
    
    def extract_strategy_patterns(self, trained_model, test_scenarios):
        """提取RL模型的决策模式"""
        patterns = {}
        
        for scenario in test_scenarios:
            state = scenario['state']
            action = trained_model.predict(state)
            
            # 分析决策逻辑
            patterns[scenario['type']] = {
                'typical_multiplier': action,
                'key_factors': self.identify_key_factors(state, action),
                'risk_assessment': self.assess_decision_risk(state, action),
                'success_probability': scenario.get('historical_success', 0.5)
            }
        
        return patterns
    
    def generate_reasoning_template(self, patterns):
        """生成推理模板"""
        template = """
        基于历史最优策略分析：
        
        情况类型：{scenario_type}
        典型做法：出价系数 {typical_multiplier:.3f}
        关键考虑因素：{key_factors}
        风险水平：{risk_level}
        历史成功率：{success_rate:.2%}
        
        建议推理逻辑：
        1. 首先评估当前市场条件是否与历史成功案例相似
        2. 考虑预算约束和时间压力的影响
        3. 评估竞争对手可能的行为模式
        4. 在风险可控的前提下优化预期收益
        """
        
        return template
```

### **3.4 第四步：交互机制设计**

**3.4.1 智能体间策略交流**
```python
class StrategyDiscussion(Event):
    """智能体策略讨论事件"""
    
    async def facilitate_discussion(self, participants):
        discussion_prompt = f"""
        当前参与讨论的广告商：{[p.profile_id for p in participants]}
        
        讨论主题：如何应对当前市场变化
        
        请各自分享：
        1. 你对当前市场的观察
        2. 你的策略调整思路
        3. 对其他广告商策略的看法
        
        注意：可以分享一般性见解，但要保护自己的核心竞争信息。
        """
        
        responses = []
        for participant in participants:
            response = await participant.generate_reaction(discussion_prompt)
            responses.append({
                'agent_id': participant.profile_id,
                'viewpoint': response,
                'timestamp': self.current_time
            })
        
        # 整理讨论结果，供各智能体参考
        return self.summarize_discussion(responses)
```

## **4. 实验设计与评估**

### **4.1 对比实验设计**

**实验组设置：**

| 实验组 | 智能体配置 | 研究目标 |
|-------|-----------|---------|
| **基准组** | 4个原始规则智能体 | 建立对比基线 |
| **RL指导组** | 2个RationalAdvertiser + 2个规则智能体 | 测试RL策略转化效果 |
| **纯LLM组** | 4个不同性格LLM智能体 | 测试纯语言推理能力 |
| **混合组** | RationalAdvertiser + 性格化智能体 + MarketAnalyst | 测试复合生态效果 |
| **交流组** | 混合组 + 策略讨论机制 | 测试智能体交流影响 |

**评估指标：**
```python
evaluation_metrics = {
    "经济效率": {
        "个体收益": "各智能体的累计利润和ROI",
        "市场效率": "总体社会福利和资源配置效率",
        "收益分布": "不同类型智能体的收益差异"
    },
    "决策质量": {
        "策略一致性": "决策推理与实际行为的一致程度",
        "适应能力": "面对市场变化的策略调整能力", 
        "学习效果": "通过交流获得的策略改进"
    },
    "推理能力": {
        "逻辑清晰度": "决策推理的逻辑性和清晰度",
        "信息利用": "对可用信息的综合利用程度",
        "创新性": "策略的新颖性和创造性"
    }
}
```

### **4.2 成果预期**

**技术贡献：**
1. **RL-LLM策略传承框架**：首次实现强化学习策略向语言模型推理的系统转化
2. **多模态决策对比**：深入分析数值优化与语言推理两种决策方式的优劣
3. **智能体交流机制**：探索语言交流对市场动态的影响机制

**学术价值：**
1. 为智能体决策机制研究提供新的实验范式
2. 为拍卖理论在AI时代的发展提供实证支持
3. 为多智能体系统中的协作与竞争研究提供新视角

**应用前景：**
1. 为实际广告拍卖系统的智能化升级提供技术参考
2. 为金融交易、供应链等其他拍卖场景提供可扩展框架
3. 为人机协作决策系统设计提供理论指导

## **5. 技术实现路线图**

### **第一周：基础框架搭建**
- **Day 1-2**: 设置YuLan-OneSim环境结构，定义场景配置文件
- **Day 3-4**: 实现基础智能体类（RationalAdvertiser, MarketAnalyst）
- **Day 5-7**: 适配GSP拍卖机制，完成基本仿真流程

### **第二周：策略传承与优化**  
- **Day 8-9**: 实现RL策略知识提取和模板生成
- **Day 10-11**: 开发智能体交流机制和讨论模块
- **Day 12-14**: 完善各类智能体的推理逻辑和决策机制

### **第三周：实验验证与优化**
- **Day 15-17**: 运行对比实验，收集数据和分析结果
- **Day 18-19**: 优化智能体策略和交互机制
- **Day 20-21**: 准备比赛材料（PPT、论文、演示系统）

## **6. 预期创新亮点**

1. **跨范式决策融合**：首次系统性地将强化学习的数值优化结果转化为大语言模型的推理依据

2. **认知多样性仿真**：通过不同推理风格的智能体模拟真实市场中的认知多样性

3. **实时策略演化**：智能体能够通过语言交流实现策略学习和适应，模拟真实交易者的社会学习过程

4. **可解释决策过程**：相比黑盒RL模型，基于语言推理的决策过程完全可解释和可审计

该整合方案既保持了原项目的技术深度，又充分利用了YuLan-OneSim平台的语言智能优势，为参赛提供了具有创新性和实用价值的研究方向。