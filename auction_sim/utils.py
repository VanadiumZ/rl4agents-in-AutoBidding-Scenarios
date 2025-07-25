# /auction_sim/utils.py
"""
Utility functions for visualization and analysis
"""
import os
import pandas as pd
import numpy as np

def generate_all_visualizations(agents):
    """
    Generate visualizations for experiment results
    """
    # Ensure results directory exists
    os.makedirs("auction_sim/results", exist_ok=True)
    
    # Simple CSV export for now
    data = []
    for agent in agents:
        total_cost = agent.get_total_cost()
        win_count = sum(1 for record in agent.history if record.get('result') and record.get('result', {}).get('won', False))
        cumulative_profit = agent.get_cumulative_profit()
        roi = agent.get_roi()
        win_rate = (win_count / len(agent.history)) * 100 if agent.history else 0
        
        data.append({
            'Agent_ID': agent.id,
            'Agent_Type': agent.__class__.__name__.replace('Agent', ''),
            'Budget_Left': agent.budget,
            'Total_Cost': total_cost,
            'Win_Count': win_count,
            'Win_Rate(%)': win_rate,
            'Cumulative_Profit': cumulative_profit,
            'ROI(%)': roi,
            'Avg_Cost_Per_Win': total_cost / win_count if win_count > 0 else 0
        })
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv("auction_sim/results/experiment_summary.csv", index=False)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    print("\nExperiment summary saved to: auction_sim/results/experiment_summary.csv")
    
    return df