#!/usr/bin/env python3
"""
Create visualizations showing relative performance compared to NearestNeighbourSolver
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def load_relative_results():
    """Load all score and runtime CSV files and calculate relative performance"""
    results_dir = "/root/Graphite-Subnet/tests/evaluation_results"
    
    # Find all score and runtime files
    score_files = glob.glob(os.path.join(results_dir, "*_scores.csv"))
    runtime_files = glob.glob(os.path.join(results_dir, "*_run_times.csv"))
    
    # Exclude summary files
    score_files = [f for f in score_files if not any(x in f for x in ['solver_scores', 'final_algorithm_summary', 'solver_relative_scores'])]
    runtime_files = [f for f in runtime_files if not any(x in f for x in ['solver_run_times'])]
    
    print(f"Found {len(score_files)} score files and {len(runtime_files)} runtime files")
    
    relative_scores = {}
    avg_runtimes = {}
    solver_names = []
    
    # Process each solver
    for score_file in score_files:
        solver_name = os.path.basename(score_file).replace('_scores.csv', '')
        runtime_file = os.path.join(results_dir, f"{solver_name}_run_times.csv")
        
        if not os.path.exists(runtime_file):
            print(f"Warning: No runtime file found for {solver_name}")
            continue
            
        try:
            # Load score data
            score_df = pd.read_csv(score_file)
            runtime_df = pd.read_csv(runtime_file)
            
            # Get solver column name (should be the solver name)
            solver_col = None
            for col in score_df.columns:
                if col not in ['problem_index', 'NearestNeighbourSolver', 'problem_size']:
                    solver_col = col
                    break
            
            if solver_col is None:
                print(f"Warning: Could not find solver column in {score_file}")
                continue
            
            # Get runtime column name
            runtime_col = None
            for col in runtime_df.columns:
                if col not in ['problem_index', 'NearestNeighbourSolver', 'problem_size']:
                    runtime_col = col
                    break
            
            if runtime_col is None:
                print(f"Warning: Could not find runtime column in {runtime_file}")
                continue
            
            # Extract data
            solver_scores = score_df[solver_col].values
            nn_scores = score_df['NearestNeighbourSolver'].values
            solver_runtimes = runtime_df[runtime_col].values
            
            # Remove inf and nan values
            valid_mask = ~(np.isinf(solver_scores) | np.isnan(solver_scores) | 
                          np.isinf(nn_scores) | np.isnan(nn_scores) |
                          np.isinf(solver_runtimes) | np.isnan(solver_runtimes))
            
            solver_scores = solver_scores[valid_mask]
            nn_scores = nn_scores[valid_mask]
            solver_runtimes = solver_runtimes[valid_mask]
            
            if len(solver_scores) == 0:
                print(f"Warning: No valid data for {solver_name}")
                continue
            
            # Calculate relative scores: mean(s1/n1, s2/n2, s3/n3, ...)
            relative_ratios = solver_scores / nn_scores
            avg_relative_score = np.mean(relative_ratios)
            
            # Calculate average runtime
            avg_runtime = np.mean(solver_runtimes)
            
            relative_scores[solver_name] = avg_relative_score
            avg_runtimes[solver_name] = avg_runtime
            solver_names.append(solver_name)
            
            print(f"{solver_name}: Relative Score = {avg_relative_score:.4f}, Avg Runtime = {avg_runtime:.2f}s")
            
        except Exception as e:
            print(f"Error processing {solver_name}: {e}")
    
    return relative_scores, avg_runtimes, solver_names

def create_relative_visualization():
    """Create visualization showing relative performance"""
    
    # Load data
    relative_scores, avg_runtimes, solver_names = load_relative_results()
    
    if len(solver_names) == 0:
        print("No valid solver data found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('TSP Solver Performance vs NearestNeighbourSolver Baseline', fontsize=16, fontweight='bold')
    
    # Sort solvers by relative score (lower is better)
    sorted_solvers = sorted(solver_names, key=lambda x: relative_scores[x])
    
    # 1. Relative Score Comparison (Bar Chart)
    ax1 = axes[0, 0]
    scores = [relative_scores[solver] for solver in sorted_solvers]
    solver_labels = [solver.replace('_', ' ').title() for solver in sorted_solvers]
    
    bars = ax1.bar(range(len(sorted_solvers)), scores, color='skyblue', alpha=0.7)
    ax1.set_title('Relative Performance (Lower is Better)', fontweight='bold')
    ax1.set_ylabel('Average Score Ratio (Solver/NN)')
    ax1.set_xlabel('Solvers')
    ax1.set_xticks(range(len(sorted_solvers)))
    ax1.set_xticklabels(solver_labels, rotation=45, ha='right')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='NearestNeighbour Baseline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Average Runtime Comparison (Bar Chart)
    ax2 = axes[0, 1]
    runtimes = [avg_runtimes[solver] for solver in sorted_solvers]
    
    bars2 = ax2.bar(range(len(sorted_solvers)), runtimes, color='lightcoral', alpha=0.7)
    ax2.set_title('Average Runtime Comparison', fontweight='bold')
    ax2.set_ylabel('Average Runtime (seconds)')
    ax2.set_xlabel('Solvers')
    ax2.set_xticks(range(len(sorted_solvers)))
    ax2.set_xticklabels(solver_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, runtime) in enumerate(zip(bars2, runtimes)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{runtime:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # 3. Score vs Runtime Scatter Plot
    ax3 = axes[1, 0]
    
    for i, solver in enumerate(sorted_solvers):
        ax3.scatter(avg_runtimes[solver], relative_scores[solver], 
                   label=solver.replace('_', ' ').title(), 
                   alpha=0.7, s=100)
    
    ax3.set_xlabel('Average Runtime (seconds)')
    ax3.set_ylabel('Relative Score (Solver/NN)')
    ax3.set_title('Performance vs Runtime Trade-off', fontweight='bold')
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='NearestNeighbour Baseline')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    for solver in sorted_solvers:
        table_data.append([
            solver.replace('_', ' ').title(),
            f"{relative_scores[solver]:.4f}",
            f"{avg_runtimes[solver]:.2f}s"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Solver', 'Rel. Score', 'Avg Runtime'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f1f1f2' if i % 2 == 0 else 'white')
    
    ax4.set_title('Performance Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "/root/Graphite-Subnet/solver_relative_performance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Relative performance visualization saved to: {output_file}")
    
    # Create summary CSV
    create_relative_summary_csv(sorted_solvers, relative_scores, avg_runtimes)
    
    plt.show()

def create_relative_summary_csv(sorted_solvers, relative_scores, avg_runtimes):
    """Create a CSV file with relative performance summary"""
    
    summary_data = []
    for solver in sorted_solvers:
        summary_data.append({
            'Solver': solver.replace('_', ' ').title(),
            'Relative_Score': relative_scores[solver],
            'Average_Runtime_Seconds': avg_runtimes[solver],
            'Performance_vs_NN': f"{((1 - relative_scores[solver]) * 100):.2f}%" if relative_scores[solver] < 1 else f"{((relative_scores[solver] - 1) * 100):.2f}% worse"
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_file = "/root/Graphite-Subnet/solver_relative_performance_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"Relative performance summary saved to: {summary_file}")
    
    print("\n" + "="*100)
    print("RELATIVE PERFORMANCE SUMMARY (vs NearestNeighbourSolver)")
    print("="*100)
    print(df_summary.to_string(index=False, float_format='%.4f'))
    print("="*100)

if __name__ == "__main__":
    # Set style
    plt.style.use('default')
    
    # Create visualizations
    create_relative_visualization()
