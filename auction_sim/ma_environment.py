# /auction_sim/ma_environment.py
"""
Multi-Agent Auction Environment for k=2 scenario
Fixed reward function to avoid negative training rewards
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any
from . import config
from .auction import GSPAuction
from .agents import Agent, TruthfulAgent, ConservativeAgent, AggressiveAgent

class MultiAgentAuctionEnv:
    """
    Multi-Agent Environment wrapper for auction simulation
    Designed for k=2 learning agents competing with rule-based agents
    """
    
    def __init__(self, learning_agent_ids: List[str], rule_agents: List[Agent]):
        self.learning_agent_ids = learning_agent_ids
        self.rule_agents = rule_agents
        self.n_learning_agents = len(learning_agent_ids)
        
        # Initialize auction
        self.auction = GSPAuction(config.N_SLOTS, config.CTR_POSITIONS, config.CTR_NOISE_STD)
        
        # Environment state
        self.current_round = 0
        self.max_rounds = config.SIMULATION_ROUNDS
        self.current_true_value = 0.0
        
        # Agent states (for learning agents)
        self.agent_budgets = {aid: config.AGENT_BUDGET for aid in learning_agent_ids}
        self.agent_histories = {aid: [] for aid in learning_agent_ids}
        self.round_history = []  # Track auction results for win rate calculation
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Performance tracking
        self.episode_rewards = {aid: 0.0 for aid in learning_agent_ids}
        
    def _setup_spaces(self):
        """Setup observation and action spaces for multi-agent learning"""
        
        # Observation space for each agent (expanded with market signals)
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Action space: continuous bid multiplier [0.5, 1.5]
        self.action_space = spaces.Box(low=0.5, high=1.5, shape=(1,), dtype=np.float32)
        
        print(f"Multi-Agent Environment initialized:")
        print(f"  - Learning agents: {self.n_learning_agents}")
        print(f"  - Rule agents: {len(self.rule_agents)}")
        print(f"  - Observation space: {self.observation_space}")
        print(f"  - Action space: {self.action_space}")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment for new episode"""
        self.current_round = 0
        self.current_true_value = 0.0
        
        # Reset agent states
        self.agent_budgets = {aid: config.AGENT_BUDGET for aid in self.learning_agent_ids}
        self.agent_histories = {aid: [] for aid in self.learning_agent_ids}
        
        # Reset rule agents
        for agent in self.rule_agents:
            agent.budget = agent.initial_budget
            agent.history = []
        
        self.episode_rewards = {aid: 0.0 for aid in self.learning_agent_ids}
        
        # Return initial observations
        return self._get_observations()
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get current observations for all learning agents"""
        observations = {}
        
        # Generate current true value and perceived values
        self.current_true_value = np.random.uniform(*config.TRUE_VALUE_RANGE)
        
        for agent_id in self.learning_agent_ids:
            # Add perception noise
            perceived_value = self.current_true_value + np.random.normal(0, config.AGENT_PERCEPTION_NOISE_STD)
            perceived_value = max(0, perceived_value)
            
            obs = self._build_observation(agent_id, perceived_value)
            observations[agent_id] = obs
            
        return observations
    
    def _build_observation(self, agent_id: str, perceived_value: float) -> np.ndarray:
        """Build observation vector for a specific learning agent"""
        
        # 1. Normalized perceived value
        norm_perceived_value = (perceived_value - config.TRUE_VALUE_RANGE[0]) / \
                              (config.TRUE_VALUE_RANGE[1] - config.TRUE_VALUE_RANGE[0])
        norm_perceived_value = np.clip(norm_perceived_value, 0.0, 1.0)
        
        # 2. Remaining budget ratio
        budget_ratio = self.agent_budgets[agent_id] / config.AGENT_BUDGET
        
        # 3. Time ratio
        time_ratio = (self.max_rounds - self.current_round) / self.max_rounds
        
        # 4. Recent win rate
        recent_wins = 0
        recent_rounds = min(len(self.agent_histories[agent_id]), 100)
        if recent_rounds > 0:
            recent_history = self.agent_histories[agent_id][-recent_rounds:]
            recent_wins = sum(1 for record in recent_history 
                            if record.get('won', False))
            recent_win_rate = recent_wins / recent_rounds
        else:
            recent_win_rate = 0.0
        
        # 5. Recent average profit (normalized)
        recent_profit = 0.0
        if recent_rounds > 0:
            recent_history = self.agent_histories[agent_id][-recent_rounds:]
            profits = [record.get('profit', 0.0) for record in recent_history]
            profits = [float(p) if p is not None else 0.0 for p in profits]
            recent_profit = np.mean(profits) if profits else 0.0
            recent_profit = np.clip(recent_profit / 10.0, -1.0, 1.0)
        
        # 6. Opponent win rate
        opponent_win_rate = 0.0
        if len(self.learning_agent_ids) > 1:
            other_agents = [aid for aid in self.learning_agent_ids if aid != agent_id]
            total_wins = 0
            total_rounds = 0
            
            for other_id in other_agents:
                other_recent = min(len(self.agent_histories[other_id]), 100)
                if other_recent > 0:
                    other_history = self.agent_histories[other_id][-other_recent:]
                    other_wins = sum(1 for record in other_history 
                                   if record.get('won', False))
                    total_wins += other_wins
                    total_rounds += other_recent
            
            opponent_win_rate = total_wins / total_rounds if total_rounds > 0 else 0.0
        
        # 7. Market competition level
        total_agents = len(self.learning_agent_ids) + len(self.rule_agents)
        competition_level = min(total_agents / 10.0, 1.0)
        
        observation = np.array([
            norm_perceived_value,
            budget_ratio,
            time_ratio,
            recent_win_rate,
            recent_profit,
            opponent_win_rate,
            competition_level
        ], dtype=np.float32)
        
        return observation
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one step in the environment"""
        self.current_round += 1
        
        # Generate perceived values for this round
        observations = self._get_observations()
        
        # Collect all bids
        all_bids = {}
        perceived_values = {}
        
        # Learning agents bids
        for agent_id in self.learning_agent_ids:
            if agent_id in actions:
                obs = observations[agent_id]
                perceived_value = obs[0] * (config.TRUE_VALUE_RANGE[1] - config.TRUE_VALUE_RANGE[0]) + config.TRUE_VALUE_RANGE[0]
                perceived_values[agent_id] = perceived_value
                
                bid_multiplier = float(actions[agent_id][0])
                bid_price = perceived_value * bid_multiplier
                
                if self.agent_budgets[agent_id] >= bid_price:
                    all_bids[agent_id] = bid_price
        
        # Rule agents bids
        for agent in self.rule_agents:
            if agent.budget > 0:
                perceived_value = agent.perceive(self.current_true_value)
                bid_price = agent.bid(perceived_value)
                
                if agent.can_afford_bid(bid_price):
                    all_bids[agent.id] = bid_price
                    perceived_values[agent.id] = perceived_value
        
        # Run auction
        auction_results = {}
        if all_bids:
            auction_results = self.auction.run_auction(all_bids)
        
        # Record round history for competitive analysis
        self.round_history.append(auction_results)
        
        # Calculate rewards and update states
        rewards = {}
        for agent_id in self.learning_agent_ids:
            reward, profit = self._calculate_reward(agent_id, auction_results, perceived_values.get(agent_id, 0))
            rewards[agent_id] = reward
            self.episode_rewards[agent_id] += reward
            
            # Update agent history
            result = auction_results.get(agent_id)
            won = result['won'] if result else False
            cost = 0.0
            
            if result and result['won']:
                expected_cost = result['cost_per_click'] * result['slot_ctr']
                cost = min(expected_cost, self.agent_budgets[agent_id])
                self.agent_budgets[agent_id] -= cost
            
            self.agent_histories[agent_id].append({
                'round': self.current_round,
                'won': won,
                'profit': float(profit) if profit is not None else 0.0,
                'cost': float(cost),
                'budget': float(self.agent_budgets[agent_id])
            })
        
        # Update rule agents
        for agent in self.rule_agents:
            result = auction_results.get(agent.id)
            if result and result['won']:
                true_value_profit = self.current_true_value * result['slot_ctr']
                expected_cost = result['cost_per_click'] * result['slot_ctr']
                profit = true_value_profit - expected_cost
            else:
                profit = 0.0
            
            agent.update(result, self.current_round)
        
        # Check termination conditions
        terminated = {agent_id: False for agent_id in self.learning_agent_ids}
        truncated = {agent_id: self.current_round >= self.max_rounds for agent_id in self.learning_agent_ids}
        
        # Info dict
        info = {agent_id: {
            'episode_reward': self.episode_rewards[agent_id],
            'budget_remaining': self.agent_budgets[agent_id],
            'round': self.current_round
        } for agent_id in self.learning_agent_ids}
        
        return observations, rewards, terminated, truncated, info
    
    def _calculate_reward(self, agent_id: str, auction_results: Dict, perceived_value: float) -> Tuple[float, float]:
        """
        ENHANCED reward function with competitive pressure and long-term optimization
        """
        result = auction_results.get(agent_id)
        
        # Calculate current round profit and cost
        if result and result['won']:
            true_profit = self.current_true_value * result['slot_ctr']
            expected_cost = result['cost_per_click'] * result['slot_ctr']
            current_profit = float(true_profit - expected_cost)
            current_cost = float(min(expected_cost, self.agent_budgets[agent_id]))
        else:
            current_profit = 0.0
            current_cost = 0.0
        
        # Base reward: immediate profit (positive when profitable)
        reward = current_profit
        
        # COMPETITIVE PRESSURE: Reward relative performance vs opponents
        learning_agents_profit = [current_profit if aid == agent_id else 0.0 for aid in self.learning_agent_ids]
        
        # Calculate rule-based agents' estimated profits for comparison
        rule_agents_profits = []
        for aid, agent_result in auction_results.items():
            if aid not in self.learning_agent_ids and agent_result and agent_result['won']:
                rule_profit = self.current_true_value * agent_result['slot_ctr'] - agent_result['cost_per_click'] * agent_result['slot_ctr']
                rule_agents_profits.append(rule_profit)
        
        if rule_agents_profits:
            avg_rule_profit = np.mean(rule_agents_profits)
            # Reward for outperforming rule-based agents
            competitive_bonus = 0.3 * np.tanh((current_profit - avg_rule_profit) / max(abs(avg_rule_profit), 1.0))
            reward += competitive_bonus
        
        # LONG-TERM ROI OPTIMIZATION
        if result and result['won'] and current_cost > 0:
            immediate_roi = (current_profit / current_cost) * 100.0
            if immediate_roi > 0:
                # Stronger bonus for high ROI wins
                efficiency_bonus = 0.4 * np.tanh(immediate_roi / 50.0)  # More aggressive scaling
                reward += efficiency_bonus
            else:
                # Penalty for unprofitable wins
                reward -= 0.2
        
        # WIN RATE INCENTIVE: Bonus for maintaining competitive win rates
        if hasattr(self, 'round_history') and len(self.round_history) > 100:
            recent_wins = sum(1 for r in self.round_history[-100:] if r.get(agent_id, {}).get('won', False))
            win_rate = recent_wins / 100.0
            target_win_rate = 0.125  # Target 12.5% win rate (fair share for 8 agents)
            
            if win_rate >= target_win_rate:
                win_bonus = 0.2 * (win_rate / target_win_rate - 1.0)
                reward += win_bonus
            else:
                # Penalty for very low win rates
                if win_rate < target_win_rate * 0.5:
                    reward -= 0.1
        
        # BUDGET EFFICIENCY: Penalize poor budget management
        budget_ratio = self.agent_budgets[agent_id] / config.AGENT_BUDGET
        time_ratio = (self.max_rounds - self.current_round) / self.max_rounds
        
        if time_ratio > 0.1:  # Don't penalize near end of auction
            ideal_budget_ratio = time_ratio * 0.8  # Should use budget more aggressively
            if budget_ratio > ideal_budget_ratio * 1.5:  # Too conservative
                reward -= 0.1 * (budget_ratio - ideal_budget_ratio)
            elif budget_ratio < ideal_budget_ratio * 0.3:  # Too aggressive
                reward -= 0.15 * (ideal_budget_ratio - budget_ratio)
        
        # Scale reward for training stability
        reward = reward / 1.5
        
        return float(reward), float(current_profit)
    
    def render(self):
        """Optional rendering for debugging"""
        print(f"Round {self.current_round}/{self.max_rounds}")
        print(f"True Value: {self.current_true_value:.2f}")
        for agent_id in self.learning_agent_ids:
            budget = self.agent_budgets[agent_id]
            episode_reward = self.episode_rewards[agent_id]
            print(f"  {agent_id}: Budget={budget:.2f}, Episode Reward={episode_reward:.2f}")