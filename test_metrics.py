#!/usr/bin/env python3
"""
Quick test to verify metrics calculation fix
"""
import sys
sys.path.append('.')

from auction_sim.hybrid_trainer import HybridTrainer

def test_metrics_fix():
    """
    Test the fixed metrics calculation with debug output
    """
    print("="*60)
    print("TESTING METRICS CALCULATION FIX")
    print("="*60)
    print("\nThis test will run a short hybrid training session")
    print("with debug output to verify metrics are calculated correctly.")
    print()
    
    # Create hybrid trainer
    hybrid_trainer = HybridTrainer(
        use_bc_pretraining=True,
        bc_episodes=3  # Very small for quick testing
    )
    
    # Run short training to test metrics
    results = hybrid_trainer.train_complete_pipeline(
        n_episodes=50  # Short test
    )
    
    print("\n" + "="*60)
    print("METRICS FIX TEST COMPLETED")
    print("="*60)
    
    curriculum_controller = results['curriculum_controller']
    final_summary = curriculum_controller.get_stage_summary()
    
    print(f"\nFinal metrics verification:")
    print(f"  Final stage: {curriculum_controller.current_stage.name}")
    print(f"  Episodes in stage: {curriculum_controller.stage_episodes}")
    print(f"  Success episodes: {curriculum_controller.success_episodes}")
    print(f"  Win rate: {final_summary.get('avg_win_rate', 0):.1%}")
    print(f"  ROI: {final_summary.get('avg_roi', 0):.1f}%")
    print(f"  Budget usage: {final_summary.get('avg_budget_usage', 0):.1%}")
    print(f"  Bid ratio: {final_summary.get('avg_bid_ratio', 1.0):.2f}")
    
    # Check if metrics look reasonable now
    metrics_ok = True
    if final_summary.get('avg_budget_usage', 0) == 0.0:
        print("\n⚠️  WARNING: Budget usage still showing 0% - may indicate budget tracking issue")
        metrics_ok = False
    
    if final_summary.get('avg_roi', 0) == 0.0 and final_summary.get('avg_win_rate', 0) > 0.1:
        print("\n⚠️  WARNING: ROI is 0% despite good win rate - may indicate profit calculation issue")
        metrics_ok = False
    
    if curriculum_controller.stage_episodes > 30 and curriculum_controller.success_episodes == 0:
        print("\n⚠️  WARNING: No success episodes after many attempts - criteria may be too strict")
        metrics_ok = False
    
    if metrics_ok:
        print("\n✅ Metrics calculation appears to be working correctly!")
    else:
        print("\n❌ Metrics calculation still has issues that need further investigation.")
    
    return results

if __name__ == "__main__":
    test_metrics_fix()