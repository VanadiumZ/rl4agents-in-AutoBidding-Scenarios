# Curriculum Learning Implementation Plan

## Overview
This document outlines the practical implementation plan for curriculum learning to address the poor performance of learning agents in the auto-bidding scenario. The current learning agents achieve only 2-3% win rates compared to 28-35% for rule-based agents, primarily due to:
- Starting with overly complex multi-objective optimization
- Sparse reward signals leading to conservative bidding
- Facing expert-level opponents from the beginning

## Implementation Phases

### Phase 0: Solo Practice
**Goal**: Learn basic auction mechanics without competition

#### Environment Configuration
```python
# Stage 0 Configuration
learning_agents = 1
rule_agents = []  # No competition
n_slots = 1
budget = 50000  # Generous budget for exploration
max_rounds = 5000  # Shorter episodes for faster learning
```

#### Reward Function
```python
def stage0_reward(won, profit, perceived_value, bid):
    if not won:
        # Penalty for not winning when alone - this shouldn't happen!
        return -1.0
    
    # Reward for winning (always should win when alone)
    base_reward = 1.0
    
    # Bonus for reasonable bidding (not too high)
    if bid < perceived_value * 1.5:
        efficiency_bonus = 0.5
    else:
        efficiency_bonus = 0.0
    
    # Small profit bonus to start learning value
    profit_bonus = np.clip(profit / 10.0, -0.5, 0.5)
    
    return base_reward + efficiency_bonus + profit_bonus
```

#### Success Criteria
- Win rate > 99% (should almost always win when alone)
- Average bid < 1.2 * perceived_value
- Episodes to advance: 100 successful episodes

#### Implementation Tasks
1. Create `CurriculumStage` enum in config.py
2. Modify `MultiAgentAuctionEnv.__init__` to accept curriculum_stage parameter
3. Implement `_stage0_reward` method
4. Add solo mode support (no rule agents)

### Phase 1: Gentle Competition
**Goal**: Learn to compete against predictable opponents

#### Environment Configuration
```python
# Stage 1 Configuration
learning_agents = 2
rule_agents = [TruthfulAgent, TruthfulAgent]  # Predictable opponents
n_slots = 2  # More winning opportunities
budget = 40000  # Still generous
max_rounds = 8000
```

#### Reward Function
```python
def stage1_reward(won, profit, perceived_value, bid, recent_win_rate):
    # Base profit reward (scaled down for stability)
    profit_reward = profit / 2.0
    
    if won:
        # Win bonus (larger than final stage)
        win_bonus = 0.1
        
        # Efficiency bonus for profitable wins
        if profit > 0:
            roi = (profit / (bid * slot_ctr)) * 100
            efficiency_bonus = 0.05 * np.tanh(roi / 50.0)
        else:
            efficiency_bonus = -0.02  # Small penalty for unprofitable wins
    else:
        win_bonus = 0.0
        efficiency_bonus = 0.0
        
        # Participation bonus for competitive bidding
        if bid > perceived_value * 0.7:
            participation_bonus = 0.02
        else:
            participation_bonus = 0.0
    
    # Competition balance bonus
    target_win_rate = 1.0 / total_agents
    if abs(recent_win_rate - target_win_rate) < 0.1:
        balance_bonus = 0.05
    else:
        balance_bonus = 0.0
    
    total_reward = profit_reward + win_bonus + efficiency_bonus + participation_bonus + balance_bonus
    return np.clip(total_reward, -1.0, 1.0)
```

#### Success Criteria
- Win rate > 40% (above fair share of 33% for 6 agents)
- ROI > 0% (at least breaking even)
- Budget usage > 60%
- Episodes to advance: 200 successful episodes

#### Implementation Tasks
1. Implement `_stage1_reward` method
2. Add curriculum-aware opponent selection
3. Implement transition logic between stages
4. Add detailed logging for curriculum metrics

## Code Structure Changes

### 1. Configuration Updates (`config.py`)
```python
from enum import Enum

class CurriculumStage(Enum):
    SOLO = 0
    GENTLE = 1
    MIXED = 2
    FULL = 3

# Curriculum-specific configurations
CURRICULUM_CONFIGS = {
    CurriculumStage.SOLO: {
        'n_learning': 1,
        'n_truthful': 0,
        'n_conservative': 0,
        'n_aggressive': 0,
        'n_slots': 1,
        'budget': 50000,
        'max_rounds': 5000
    },
    CurriculumStage.GENTLE: {
        'n_learning': 2,
        'n_truthful': 2,
        'n_conservative': 0,
        'n_aggressive': 0,
        'n_slots': 2,
        'budget': 40000,
        'max_rounds': 8000
    }
}
```

### 2. Environment Modifications (`ma_environment.py`)
```python
class MultiAgentAuctionEnv:
    def __init__(self, learning_agent_ids, rule_agents, curriculum_stage=CurriculumStage.FULL):
        self.curriculum_stage = curriculum_stage
        self.stage_config = CURRICULUM_CONFIGS.get(curriculum_stage, {})
        
        # Override config based on curriculum stage
        if self.stage_config:
            self.n_slots = self.stage_config['n_slots']
            self.max_rounds = self.stage_config['max_rounds']
            # ... other overrides
        
    def _calculate_reward(self, agent_id, auction_results, perceived_value):
        if self.curriculum_stage == CurriculumStage.SOLO:
            return self._stage0_reward(agent_id, auction_results, perceived_value)
        elif self.curriculum_stage == CurriculumStage.GENTLE:
            return self._stage1_reward(agent_id, auction_results, perceived_value)
        else:
            return self._full_reward(agent_id, auction_results, perceived_value)
```

### 3. Curriculum Controller (`curriculum_controller.py`)
```python
class CurriculumController:
    def __init__(self):
        self.current_stage = CurriculumStage.SOLO
        self.stage_episodes = 0
        self.stage_metrics = defaultdict(list)
        self.success_episodes = 0
        
    def update_metrics(self, episode_stats):
        """Update metrics for current stage"""
        self.stage_episodes += 1
        
        # Track key metrics
        self.stage_metrics['win_rates'].append(episode_stats['avg_win_rate'])
        self.stage_metrics['roi'].append(episode_stats['avg_roi'])
        self.stage_metrics['budget_usage'].append(episode_stats['budget_usage'])
        
        # Check if episode was successful
        if self.check_episode_success(episode_stats):
            self.success_episodes += 1
    
    def should_advance(self):
        """Check if ready to advance to next stage"""
        if self.current_stage == CurriculumStage.SOLO:
            return self.success_episodes >= 100
        elif self.current_stage == CurriculumStage.GENTLE:
            return self.success_episodes >= 200
        return False
    
    def advance_stage(self):
        """Move to next curriculum stage"""
        stages = list(CurriculumStage)
        current_idx = stages.index(self.current_stage)
        if current_idx < len(stages) - 1:
            self.current_stage = stages[current_idx + 1]
            self.reset_stage_metrics()
            return True
        return False
```

### 4. Training Loop Modifications (`ma_trainer.py`)
```python
class MAPPOTrainer:
    def __init__(self, ..., use_curriculum=True):
        self.use_curriculum = use_curriculum
        if use_curriculum:
            self.curriculum_controller = CurriculumController()
            
    def train(self):
        for episode in range(self.n_episodes):
            # Get current curriculum stage
            if self.use_curriculum:
                stage = self.curriculum_controller.current_stage
                env = self.create_env_for_stage(stage)
            
            # Run episode
            episode_stats = self.run_episode(env)
            
            # Update curriculum
            if self.use_curriculum:
                self.curriculum_controller.update_metrics(episode_stats)
                if self.curriculum_controller.should_advance():
                    print(f"Advancing to next curriculum stage!")
                    self.curriculum_controller.advance_stage()
```

## Experiment Plan

### Phase 0 Experiments
1. **Baseline Solo Performance**
   - Run 500 episodes with solo learner
   - Verify 99%+ win rate
   - Analyze bidding patterns

2. **Reward Shaping Validation**
   - Test different reward weights
   - Ensure agent learns to bid reasonably (not too high)

### Phase 1 Experiments
1. **Transition Testing**
   - Smooth transition from Stage 0 to Stage 1
   - Performance shouldn't collapse

2. **Competition Learning**
   - Verify agent learns to compete with TruthfulAgents
   - Check if win rate stabilizes around fair share

3. **Hyperparameter Tuning**
   - Learning rate adjustments for new reward scale
   - Exploration vs exploitation balance

## Success Metrics

### Overall Success Indicators
1. **Learning Efficiency**: Episodes to reach success criteria per stage
2. **Performance Stability**: Variance in metrics after convergence
3. **Transfer Success**: Performance retention when advancing stages
4. **Final Performance**: Comparison with rule-based agents after full curriculum

### Key Performance Indicators (KPIs)
- Win Rate Improvement: From 3% → 20%+ 
- Budget Utilization: From 2% → 70%+
- ROI Achievement: From negative → 10%+
- Training Time: Target < 10,000 total episodes

## Implementation Timeline

### Week 1: Foundation
- Day 1-2: Implement Stage 0 environment modifications
- Day 3-4: Create curriculum controller and metrics tracking
- Day 5: Initial testing and debugging

### Week 2: Refinement
- Day 1-2: Implement Stage 1 and transition logic
- Day 3-4: Run full experiments for Stage 0 and 1
- Day 5: Analysis and parameter tuning

### Week 3: Validation
- Day 1-2: Extended experiments with multiple seeds
- Day 3-4: Implement visualization and analysis tools
- Day 5: Prepare for Stage 2 and beyond

## Risk Mitigation

### Potential Issues and Solutions
1. **Stage Transition Collapse**
   - Solution: Implement gradual transition with mixed rewards
   - Fallback: Allow temporary performance dip with recovery period

2. **Overfitting to Stage**
   - Solution: Add noise and variation within each stage
   - Validation: Test on slightly different configurations

3. **Slow Learning**
   - Solution: Adjust learning rates per stage
   - Alternative: Use pretrained networks from previous stages

## Next Steps
1. Implement Stage 0 environment modifications
2. Create basic curriculum controller
3. Run initial experiments
4. Iterate based on results