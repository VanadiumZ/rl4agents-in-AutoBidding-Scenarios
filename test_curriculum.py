#!/usr/bin/env python3
"""
Test script for curriculum learning implementation
Tests Stage 0 (Solo) and Stage 1 (Gentle Competition)
"""
import sys
sys.path.append('.')

from auction_sim.ma_runner import train_multi_agent
from auction_sim.config import CurriculumStage

def test_curriculum_learning():
    """
    Test curriculum learning with a small number of episodes
    """
    print("="*70)
    print("TESTING CURRICULUM LEARNING IMPLEMENTATION")
    print("="*70)
    print("\nThis test will run Stage 0 (Solo) and Stage 1 (Gentle Competition)")
    print("with a reduced number of episodes for quick validation.\n")
    
    # Run training with curriculum learning
    # Using fewer episodes for testing (normally would use 1000+)
    n_episodes = 150  # Enough to potentially complete Stage 0 and start Stage 1
    
    trainer, learning_agents, rule_agents = train_multi_agent(
        n_episodes=n_episodes,
        save_models=True,
        use_curriculum=True
    )
    
    print("\n" + "="*70)
    print("CURRICULUM LEARNING TEST COMPLETED")
    print("="*70)
    
    # Print final statistics
    if hasattr(trainer, 'curriculum_controller') and trainer.curriculum_controller:
        controller = trainer.curriculum_controller
        print(f"\nFinal curriculum stage: {controller.current_stage.name}")
        print(f"Total episodes in current stage: {controller.stage_episodes}")
        print(f"Success episodes in current stage: {controller.success_episodes}")
        print(f"\nFinal stage summary:")
        print(controller.get_stage_summary())
        
        if controller.stage_history:
            print(f"\nCompleted stages:")
            for stage_info in controller.stage_history:
                print(f"  - {stage_info['stage'].name}: {stage_info['episodes']} episodes")
    
    return trainer, learning_agents, rule_agents

def test_without_curriculum():
    """
    Test regular training without curriculum for comparison
    """
    print("\n" + "="*70)
    print("TESTING REGULAR TRAINING (NO CURRICULUM)")
    print("="*70)
    
    # Run training without curriculum learning
    n_episodes = 50  # Fewer episodes for quick test
    
    trainer, learning_agents, rule_agents = train_multi_agent(
        n_episodes=n_episodes,
        save_models=False,
        use_curriculum=False
    )
    
    print("\n" + "="*70)
    print("REGULAR TRAINING TEST COMPLETED")
    print("="*70)
    
    return trainer, learning_agents, rule_agents

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test curriculum learning implementation")
    parser.add_argument("--mode", choices=["curriculum", "regular", "both"], 
                       default="curriculum",
                       help="Test mode: curriculum learning, regular training, or both")
    parser.add_argument("--episodes", type=int, default=150,
                       help="Number of training episodes (default: 150)")
    
    args = parser.parse_args()
    
    if args.mode in ["curriculum", "both"]:
        print("Running curriculum learning test...")
        test_curriculum_learning()
    
    if args.mode in ["regular", "both"]:
        print("\nRunning regular training test...")
        test_without_curriculum()
    
    print("\nTest completed successfully!")