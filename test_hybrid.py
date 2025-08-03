#!/usr/bin/env python3
"""
Test script for the complete Hybrid BC + Curriculum Learning approach
"""
import sys
sys.path.append('.')

from auction_sim.hybrid_trainer import HybridTrainer

def test_hybrid_approach():
    """
    Test the complete BC + Symmetric Curriculum Learning pipeline
    """
    print("="*80)
    print("TESTING HYBRID BC + CURRICULUM LEARNING APPROACH")
    print("先模仿，后进阶 (Imitate First, Then Advance)")
    print("="*80)
    print("\\nThis test will run:")
    print("1. BC data collection from AggressiveAgent")
    print("2. BC pre-training for imitation learning")
    print("3. Symmetric curriculum learning with BC initialization")
    print("4. All stages: COOPERATIVE → COMPETITIVE → ADVANCED → FULL_SPECTRUM")
    print()
    
    # Create hybrid trainer
    hybrid_trainer = HybridTrainer(
        use_bc_pretraining=True,
        bc_episodes=8  # Reduced for quick testing
    )
    
    # Run complete pipeline
    results = hybrid_trainer.train_complete_pipeline(
        n_episodes=200  # Reduced for testing
    )
    
    print("\\n" + "="*80)
    print("HYBRID APPROACH TEST COMPLETED")
    print("="*80)
    
    # Print final results
    bc_metrics = results['bc_metrics']
    curriculum_controller = results['curriculum_controller']
    
    if bc_metrics:
        print(f"\\nBC Pre-training Results:")
        print(f"  Action Accuracy: {bc_metrics['action_accuracy_10pct']:.1%}")
        print(f"  RMSE: {bc_metrics['rmse']:.4f}")
        print(f"  Mean prediction vs target: {bc_metrics['mean_prediction']:.3f} vs {bc_metrics['mean_target']:.3f}")
    
    print(f"\\nCurriculum Learning Results:")
    print(f"  Final stage: {curriculum_controller.current_stage.name}")
    print(f"  Total episodes in current stage: {curriculum_controller.stage_episodes}")
    
    if curriculum_controller.stage_history:
        print(f"\\nCompleted curriculum stages:")
        for stage_info in curriculum_controller.stage_history:
            print(f"  - {stage_info['stage'].name}: {stage_info['episodes']} episodes")
            metrics = stage_info['final_metrics']
            print(f"    Win Rate: {metrics.get('avg_win_rate', 0):.1%}, "
                  f"ROI: {metrics.get('avg_roi', 0):.1f}%, "
                  f"Budget Usage: {metrics.get('avg_budget_usage', 0):.1%}")
    
    final_summary = curriculum_controller.get_stage_summary()
    print(f"\\nFinal stage summary:")
    print(f"  Win Rate: {final_summary.get('avg_win_rate', 0):.1%}")
    print(f"  Budget Usage: {final_summary.get('avg_budget_usage', 0):.1%}")
    print(f"  Episodes: {final_summary.get('episodes', 0)}")
    
    return results

def test_comparison():
    """
    Optional: Test BC vs No-BC comparison
    """
    print("\\n" + "="*60)
    print("COMPARISON: WITH BC vs WITHOUT BC")
    print("="*60)
    
    print("\\nTesting WITHOUT BC pre-training...")
    hybrid_no_bc = HybridTrainer(
        use_bc_pretraining=False,
        bc_episodes=0
    )
    
    results_no_bc = hybrid_no_bc.train_complete_pipeline(n_episodes=100)
    
    print("\\nComparison completed!")
    print("Check the training curves to see the difference between BC and non-BC approaches.")
    
    return results_no_bc

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test hybrid BC + curriculum learning")
    parser.add_argument("--mode", choices=["hybrid", "comparison", "both"], 
                       default="hybrid",
                       help="Test mode")
    parser.add_argument("--episodes", type=int, default=200,
                       help="Number of curriculum episodes")
    
    args = parser.parse_args()
    
    if args.mode in ["hybrid", "both"]:
        print("Running hybrid approach test...")
        results = test_hybrid_approach()
    
    if args.mode in ["comparison", "both"]:
        print("\\nRunning comparison test...")
        results_comparison = test_comparison()
    
    print("\\nAll tests completed successfully!")
    print("Check the following files for results:")
    print("  - auction_sim/results/bc_training_curves.png")
    print("  - auction_sim/results/hybrid_training_curves.png")
    print("  - auction_sim/models/hybrid_best/ (best models)")
    print("  - auction_sim/models/hybrid_final/ (final models)")