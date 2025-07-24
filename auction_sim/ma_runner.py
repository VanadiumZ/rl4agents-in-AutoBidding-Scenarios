# /auction_sim/ma_runner.py
"""
Multi-Agent Training Runner for k=2 scenario
Trains 2 learning agents using MAPPO against rule-based opponents
"""
import numpy as np
import random
import torch
from tqdm import tqdm
import os
from . import config
from .ma_environment import MultiAgentAuctionEnv
from .ma_trainer import MAPPOTrainer
from .agents import TruthfulAgent, ConservativeAgent, AggressiveAgent, MultiAgentLearningAgent
from .utils import generate_all_visualizations

def create_rule_agents():
    """Create rule-based opponents for training"""
    rule_agents = []
    
    # Create agents based on config
    agent_id_counter = 0
    
    for spec in config.EXPERIMENT_SETUP['agents']:
        if spec['type'] != 'Learning':  # Skip learning agents
            for _ in range(spec['count']):
                agent_id = f"{spec['type']}_{agent_id_counter}"
                agent_id_counter += 1
                
                if spec['type'] == 'Truthful':
                    agent = TruthfulAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD)
                elif spec['type'] == 'Conservative':
                    agent = ConservativeAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, config.SIMULATION_ROUNDS)
                elif spec['type'] == 'Aggressive':
                    total_agents = sum(s['count'] for s in config.EXPERIMENT_SETUP['agents'])
                    agent = AggressiveAgent(agent_id, spec['budget'], config.AGENT_PERCEPTION_NOISE_STD, total_agents)
                
                rule_agents.append(agent)
    
    return rule_agents

def create_learning_agents():
    """Create learning agents with proper integration"""
    learning_agents = []
    learning_agent_ids = []
    
    # Count learning agents in config
    learning_count = 0
    for spec in config.EXPERIMENT_SETUP['agents']:
        if spec['type'] == 'Learning':
            learning_count = spec['count']
            break
    
    for i in range(learning_count):
        agent_id = f"Learning_{i}"
        agent = MultiAgentLearningAgent(
            agent_id, 
            config.AGENT_BUDGET, 
            config.AGENT_PERCEPTION_NOISE_STD,
            model=None,  # Will be set by trainer
            is_training=True
        )
        learning_agents.append(agent)
        learning_agent_ids.append(agent_id)
    
    return learning_agents, learning_agent_ids

def train_multi_agent(n_episodes: int = 200, save_models: bool = True):  # Reduce episodes since each is now 16k steps
    """
    Main training function for multi-agent scenario
    """
    print("="*60)
    print("MULTI-AGENT TRAINING (k=2)")
    print("="*60)
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create agents
    rule_agents = create_rule_agents()
    learning_agents, learning_agent_ids = create_learning_agents()
    
    print(f"Created {len(rule_agents)} rule-based agents")
    print(f"Created {len(learning_agents)} learning agents")
    
    # Create environment
    env = MultiAgentAuctionEnv(learning_agent_ids, rule_agents)
    
    # Create trainer with improved hyperparameters
    trainer = MAPPOTrainer(
        obs_dim=7,
        action_dim=1, 
        n_agents=len(learning_agents),
        lr=1e-4,  # Lower learning rate for more stable learning
        gamma=0.95,  # Slightly lower discount factor
        gae_lambda=0.9,  # Lower GAE lambda
        clip_ratio=0.1,  # Smaller clip ratio for more conservative updates
        vf_coef=0.5,
        ent_coef=0.02,  # Higher entropy for more exploration
        max_grad_norm=0.3  # Smaller gradient clipping
    )
    
    # Connect models to agents
    for i, agent in enumerate(learning_agents):
        agent_id = f"Learning_{i}"
        
        # Create a simple model wrapper for the agent
        class ModelWrapper:
            def __init__(self, network, trainer, agent_id):
                self.network = network
                self.trainer = trainer
                self.agent_id = agent_id
            
            def predict(self, obs, deterministic=False):
                action, _ = self.trainer.get_action(self.agent_id, obs, deterministic)
                return np.array([action]), None
        
        model_wrapper = ModelWrapper(trainer.networks[agent_id], trainer, agent_id)
        agent.set_model(model_wrapper)
    
    # Training loop
    print(f"Starting training for {n_episodes} episodes...")
    
    best_avg_reward = float('-inf')
    episode_rewards_history = []
    
    for episode in tqdm(range(n_episodes), desc="Training Episodes"):
        # Use full simulation rounds for proper training-competition alignment
        max_steps = config.SIMULATION_ROUNDS  # Always use full 16,000 rounds
            
        episode_rewards, episode_wins = trainer.train_episode(env, max_steps=max_steps)
        
        # Track progress
        avg_reward = np.mean(list(episode_rewards.values()))
        episode_rewards_history.append(avg_reward)
        
        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            if save_models:
                trainer.save_models("auction_sim/models/best")
        
        # Logging
        if episode % 10 == 0:  # More frequent logging
            recent_avg = np.mean(episode_rewards_history[-10:]) if len(episode_rewards_history) >= 10 else avg_reward
            win_rates = {aid: (episode_wins[aid] / max_steps) for aid in learning_agent_ids}
            
            print(f"\nEpisode {episode} (full {max_steps} rounds):")
            print(f"  Average reward: {recent_avg:.2f}")
            print(f"  Episode rewards: {episode_rewards}")
            print(f"  Episode wins: {episode_wins}")
            print(f"  Win rates: {win_rates}")
            print(f"  Best avg reward: {best_avg_reward:.2f}")
    
    # Save final models
    if save_models:
        trainer.save_models("auction_sim/models/final")
        trainer.plot_training_curves("auction_sim/results/ma_training_curves.png")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    
    return trainer, learning_agents, rule_agents

def evaluate_trained_agents(trainer: MAPPOTrainer, n_eval_episodes: int = 5):
    """
    Evaluate trained agents against rule-based opponents
    """
    print("\n" + "="*50)
    print("EVALUATING TRAINED AGENTS")
    print("="*50)
    
    # Create evaluation environment
    rule_agents = create_rule_agents()
    learning_agents, learning_agent_ids = create_learning_agents()
    
    # Set models to evaluation mode
    for i, agent in enumerate(learning_agents):
        agent_id = f"Learning_{i}"
        agent.is_training = False
        
        class EvalModelWrapper:
            def __init__(self, network, trainer, agent_id):
                self.network = network
                self.trainer = trainer
                self.agent_id = agent_id
            
            def predict(self, obs, deterministic=True):
                action, _ = self.trainer.get_action(self.agent_id, obs, deterministic=True)
                return np.array([action]), None
        
        eval_wrapper = EvalModelWrapper(trainer.networks[agent_id], trainer, agent_id)
        agent.set_model(eval_wrapper)
    
    env = MultiAgentAuctionEnv(learning_agent_ids, rule_agents)
    
    # Run evaluation episodes
    all_rewards = []
    all_wins = []
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in learning_agent_ids}
        episode_wins = {agent_id: 0 for agent_id in learning_agent_ids}
        
        for step in range(config.SIMULATION_ROUNDS):
            actions = {}
            
            for agent_id in learning_agent_ids:
                action, _ = trainer.get_action(agent_id, obs[agent_id], deterministic=True)
                actions[agent_id] = np.array([action])
            
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            for agent_id in learning_agent_ids:
                episode_rewards[agent_id] += rewards[agent_id]
                if step < len(env.agent_histories[agent_id]) and env.agent_histories[agent_id][-1]['won']:
                    episode_wins[agent_id] += 1
            
            obs = next_obs
            
            if any(terminated.values()) or any(truncated.values()):
                break
        
        all_rewards.append(episode_rewards)
        all_wins.append(episode_wins)
        
        print(f"Eval Episode {episode + 1}: Rewards = {episode_rewards}, Wins = {episode_wins}")
    
    # Calculate statistics
    avg_rewards = {}
    avg_win_rates = {}
    
    for agent_id in learning_agent_ids:
        rewards = [ep_rewards[agent_id] for ep_rewards in all_rewards]
        wins = [ep_wins[agent_id] for ep_wins in all_wins]
        
        avg_rewards[agent_id] = np.mean(rewards)
        avg_win_rates[agent_id] = np.mean(wins) / config.SIMULATION_ROUNDS
    
    print(f"\nEvaluation Results (averaged over {n_eval_episodes} episodes):")
    print(f"Average Rewards: {avg_rewards}")
    print(f"Average Win Rates: {avg_win_rates}")
    
    return avg_rewards, avg_win_rates

def run_full_experiment():
    """
    Run complete multi-agent experiment: train, evaluate, and visualize
    """
    # Ensure results directory exists
    os.makedirs("auction_sim/results", exist_ok=True)
    os.makedirs("auction_sim/models", exist_ok=True)
    
    # Training phase
    trainer, learning_agents, rule_agents = train_multi_agent(n_episodes=100)  # Reduced since each episode is now full 16k rounds
    
    # Evaluation phase
    avg_rewards, avg_win_rates = evaluate_trained_agents(trainer, n_eval_episodes=3)
    
    # Create agents for final simulation with trained models
    print("\n" + "="*50)
    print("RUNNING FINAL SIMULATION")
    print("="*50)
    
    # Simply use the rule agents from the evaluation phase (they have full history)
    # For learning agents, we'll run a quick simulation to populate their history
    
    # Get learning agent IDs from the training
    learning_agent_ids = [agent.id for agent in learning_agents]
    
    env = MultiAgentAuctionEnv(learning_agent_ids, rule_agents)
    obs = env.reset()
    
    # Set models to evaluation mode for the simulation
    for i, agent in enumerate(learning_agents):
        agent_id = f"Learning_{i}"
        agent.is_training = False
    
    print("Running final simulation to generate agent histories...")
    for step in tqdm(range(config.SIMULATION_ROUNDS), desc="Final Simulation"):
        actions = {}
        
        # Get actions for all learning agents
        for agent_id in learning_agent_ids:
            action, _ = trainer.get_action(agent_id, obs[agent_id], deterministic=True)
            actions[agent_id] = np.array([action])
        
        # Environment step
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        obs = next_obs
        
        # Check if episode is done
        if any(terminated.values()) or any(truncated.values()):
            break
    
    # Create final agents with populated histories from the simulation
    final_learning_agents = []
    for i, agent_id in enumerate(learning_agent_ids):
        # Create a new agent and copy the history from the environment
        agent = MultiAgentLearningAgent(
            agent_id,
            config.AGENT_BUDGET,
            config.AGENT_PERCEPTION_NOISE_STD,
            is_training=False
        )
        
        # Copy history from environment agent histories
        agent.history = env.agent_histories[agent_id].copy()
        agent.budget = env.agent_budgets[agent_id]
        
        final_learning_agents.append(agent)
    
    # Update rule agents with their histories from the simulation
    updated_rule_agents = []
    for rule_agent in rule_agents:
        # Rule agents already have their histories updated during the simulation
        updated_rule_agents.append(rule_agent)
    
    # Combine all agents for visualization
    all_agents = final_learning_agents + updated_rule_agents
    
    # Generate visualizations
    generate_all_visualizations(all_agents)
    
    print("\n✅ Multi-agent experiment completed successfully!")
    print("Check auction_sim/results/ for visualizations and models")

if __name__ == "__main__":
    run_full_experiment()