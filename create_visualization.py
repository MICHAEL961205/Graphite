#!/usr/bin/env python3
"""
Create comprehensive visualizations of TSP solver performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path

def load_all_results():
    """Load all score and runtime CSV files"""
    results_dir = "/root/Graphite-Subnet/tests/evaluation_results"
    
    # Find all score and runtime files
    score_files = glob.glob(os.path.join(results_dir, "*_scores.csv"))
    runtime_files = glob.glob(os.path.join(results_dir, "*_run_times.csv"))
    
    # Exclude summary files
    score_files = [f for f in score_files if not any(x in f for x in ['solver_scores', 'final_algorithm_summary', 'solver_relative_scores'])]
    runtime_files = [f for f in runtime_files if not any(x in f for x in ['solver_run_times'])]
    
    print(f"Found {len(score_files)} score files and {len(runtime_files)} runtime files")
    
    all_scores = {}
    all_runtimes = {}
    
    # Load scores
    for file in score_files:
        solver_name = os.path.basename(file).replace('_scores.csv', '')
        try:
            df = pd.read_csv(file)
            if 'score' in df.columns:
                all_scores[solver_name] = df['score'].values
            elif 'Score' in df.columns:
                all_scores[solver_name] = df['Score'].values
            else:
                # Try to find any numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    all_scores[solver_name] = df[numeric_cols[0]].values
                else:
                    print(f"Warning: No numeric columns found in {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Load runtimes
    for file in runtime_files:
        solver_name = os.path.basename(file).replace('_run_times.csv', '')
        try:
            df = pd.read_csv(file)
            if 'runtime' in df.columns:
                all_runtimes[solver_name] = df['runtime'].values
            elif 'Runtime' in df.columns:
                all_runtimes[solver_name] = df['Runtime'].values
            elif 'time' in df.columns:
                all_runtimes[solver_name] = df['time'].values
            else:
                # Try to find any numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    all_runtimes[solver_name] = df[numeric_cols[0]].values
                else:
                    print(f"Warning: No numeric columns found in {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return all_scores, all_runtimes

def create_comprehensive_visualization():
    """Create comprehensive visualization of all solver results"""
    
    # Load data
    all_scores, all_runtimes = load_all_results()
    
    # Filter out solvers with no data or inf values
    valid_solvers = []
    for solver in all_scores:
        if solver in all_runtimes and len(all_scores[solver]) > 0:
            # Check if solver has valid (non-inf) data
            scores = all_scores[solver]
            runtimes = all_runtimes[solver]
            
            # Remove inf values
            valid_scores = scores[~np.isinf(scores)]
            valid_runtimes = runtimes[~np.isinf(runtimes)]
            
            if len(valid_scores) > 0 and len(valid_runtimes) > 0:
                valid_solvers.append(solver)
                all_scores[solver] = valid_scores
                all_runtimes[solver] = valid_runtimes
    
    print(f"Valid solvers: {len(valid_solvers)}")
    print(f"Solver names: {valid_solvers}")
    
    if len(valid_solvers) == 0:
        print("No valid solver data found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('TSP Solver Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Score comparison (box plot)
    ax1 = axes[0, 0]
    score_data = []
    score_labels = []
    
    for solver in valid_solvers:
        scores = all_scores[solver]
        # Remove any remaining inf or nan values
        clean_scores = scores[~np.isinf(scores) & ~np.isnan(scores)]
        if len(clean_scores) > 0:
            score_data.append(clean_scores)
            score_labels.append(solver.replace('_', ' ').title())
    
    if score_data:
        bp1 = ax1.boxplot(score_data, labels=score_labels, patch_artist=True)
        ax1.set_title('Score Distribution by Solver', fontweight='bold')
        ax1.set_ylabel('Score (Lower is Better)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp1['boxes'])))
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
    
    # 2. Runtime comparison (box plot)
    ax2 = axes[0, 1]
    runtime_data = []
    runtime_labels = []
    
    for solver in valid_solvers:
        runtimes = all_runtimes[solver]
        # Remove any remaining inf or nan values
        clean_runtimes = runtimes[~np.isinf(runtimes) & ~np.isnan(runtimes)]
        if len(clean_runtimes) > 0:
            runtime_data.append(clean_runtimes)
            runtime_labels.append(solver.replace('_', ' ').title())
    
    if runtime_data:
        bp2 = ax2.boxplot(runtime_data, labels=runtime_labels, patch_artist=True)
        ax2.set_title('Runtime Distribution by Solver', fontweight='bold')
        ax2.set_ylabel('Runtime (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp2['boxes'])))
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
    
    # 3. Score vs Runtime scatter plot
    ax3 = axes[1, 0]
    
    for i, solver in enumerate(valid_solvers):
        scores = all_scores[solver]
        runtimes = all_runtimes[solver]
        
        # Ensure same length
        min_len = min(len(scores), len(runtimes))
        scores = scores[:min_len]
        runtimes = runtimes[:min_len]
        
        # Remove inf/nan values
        valid_mask = ~(np.isinf(scores) | np.isnan(scores) | np.isinf(runtimes) | np.isnan(runtimes))
        clean_scores = scores[valid_mask]
        clean_runtimes = runtimes[valid_mask]
        
        if len(clean_scores) > 0:
            ax3.scatter(clean_runtimes, clean_scores, 
                       label=solver.replace('_', ' ').title(), 
                       alpha=0.7, s=50)
    
    ax3.set_xlabel('Runtime (seconds)')
    ax3.set_ylabel('Score (Lower is Better)')
    ax3.set_title('Score vs Runtime Trade-off', fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Average performance comparison
    ax4 = axes[1, 1]
    
    avg_scores = []
    avg_runtimes = []
    solver_names = []
    
    for solver in valid_solvers:
        scores = all_scores[solver]
        runtimes = all_runtimes[solver]
        
        # Calculate averages, excluding inf/nan
        clean_scores = scores[~np.isinf(scores) & ~np.isnan(scores)]
        clean_runtimes = runtimes[~np.isinf(runtimes) & ~np.isnan(runtimes)]
        
        if len(clean_scores) > 0 and len(clean_runtimes) > 0:
            avg_scores.append(np.mean(clean_scores))
            avg_runtimes.append(np.mean(clean_runtimes))
            solver_names.append(solver.replace('_', ' ').title())
    
    if avg_scores and avg_runtimes:
        # Create bar chart
        x_pos = np.arange(len(solver_names))
        width = 0.35
        
        bars1 = ax4.bar(x_pos - width/2, avg_scores, width, label='Avg Score', alpha=0.8)
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x_pos + width/2, avg_runtimes, width, label='Avg Runtime', alpha=0.8, color='orange')
        
        ax4.set_xlabel('Solvers')
        ax4.set_ylabel('Average Score (Lower is Better)', color='blue')
        ax4_twin.set_ylabel('Average Runtime (seconds)', color='orange')
        ax4.set_title('Average Performance Comparison', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(solver_names, rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax4_twin.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "/root/Graphite-Subnet/solver_performance_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    # Also create a summary table
    create_summary_table(valid_solvers, all_scores, all_runtimes)
    
    plt.show()

def create_summary_table(valid_solvers, all_scores, all_runtimes):
    """Create a summary table of solver performance"""
    
    summary_data = []
    
    for solver in valid_solvers:
        scores = all_scores[solver]
        runtimes = all_runtimes[solver]
        
        # Calculate statistics
        clean_scores = scores[~np.isinf(scores) & ~np.isnan(scores)]
        clean_runtimes = runtimes[~np.isinf(runtimes) & ~np.isnan(runtimes)]
        
        if len(clean_scores) > 0 and len(clean_runtimes) > 0:
            summary_data.append({
                'Solver': solver.replace('_', ' ').title(),
                'Avg Score': np.mean(clean_scores),
                'Min Score': np.min(clean_scores),
                'Max Score': np.max(clean_scores),
                'Std Score': np.std(clean_scores),
                'Avg Runtime': np.mean(clean_runtimes),
                'Min Runtime': np.min(clean_runtimes),
                'Max Runtime': np.max(clean_runtimes),
                'Std Runtime': np.std(clean_runtimes),
                'Samples': len(clean_scores)
            })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('Avg Score')
        
        # Save summary table
        summary_file = "/root/Graphite-Subnet/solver_performance_summary.csv"
        df_summary.to_csv(summary_file, index=False)
        print(f"Summary table saved to: {summary_file}")
        
        print("\n" + "="*80)
        print("SOLVER PERFORMANCE SUMMARY")
        print("="*80)
        print(df_summary.to_string(index=False, float_format='%.2f'))
        print("="*80)

if __name__ == "__main__":
    # Set style
    plt.style.use('default')
    
    # Create visualizations
    create_comprehensive_visualization()